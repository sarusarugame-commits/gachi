import os
import datetime
import time
import sqlite3
import concurrent.futures
import threading
import sys
import requests as std_requests
import json

# scraperからは2連単結果取得用の scrape_result と 2連単オッズ用の get_odds_2t をインポート
from scraper import scrape_race_data, get_session, get_odds_map, get_odds_2t, scrape_result
# predict_boatは最新の独立設定版を使用
from predict_boat import predict_race, attach_reason, load_model, filter_and_sort_bets

DB_FILE = "race_data.db"
PLACE_NAMES = {i: n for i, n in enumerate(["","桐生","戸田","江戸川","平和島","多摩川","浜名湖","蒲郡","常滑","津","三国","びわこ","住之江","尼崎","鳴門","丸亀","児島","宮島","徳山","下関","若松","芦屋","福岡","唐津","大村"])}
JST = datetime.timezone(datetime.timedelta(hours=9), 'JST')

sys.stdout.reconfigure(encoding='utf-8')

DB_LOCK = threading.Lock()
STATS = {"scanned": 0, "hits": 0, "errors": 0, "skipped": 0, "vetted": 0}
STATS_LOCK = threading.Lock()
FINISHED_RACES = set()
FINISHED_RACES_LOCK = threading.Lock()

def log(msg):
    print(f"[{datetime.datetime.now(JST).strftime('%H:%M:%S')}] {msg}", flush=True)

def error_log(msg):
    print(f"[{datetime.datetime.now(JST).strftime('%H:%M:%S')}] ❌ {msg}", file=sys.stderr, flush=True)

def send_discord(content):
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not url: return
    try:
        std_requests.post(url, json={"content": content}, timeout=10)
    except: pass

def init_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS history (
            race_id TEXT PRIMARY KEY,
            date TEXT,
            place TEXT,
            race_no INTEGER,
            predict_combo TEXT,
            status TEXT,
            profit INTEGER,
            odds REAL,
            prob REAL,
            ev REAL,
            comment TEXT,
            ticket_type TEXT
        )
    """)
    conn.close()

def report_worker(stop_event):
    log("ℹ️ レポート監視スレッド起動 (2連単/3連単 両対応)")
    while not stop_event.is_set():
        try:
            with DB_LOCK:
                conn = sqlite3.connect(DB_FILE)
                conn.row_factory = sqlite3.Row
                pending = conn.execute("SELECT * FROM history WHERE status='PENDING'").fetchall()
                sess = get_session()
                
                for p in pending:
                    try:
                        # race_id形式: 20260205_JCD_RNO_COMBO_TYPE
                        parts = p['race_id'].split('_')
                        jcd = int(parts[1])
                    except: continue
                    
                    res = scrape_result(sess, jcd, p['race_no'], p['date'])
                    if not res: continue

                    combo = p['predict_combo']
                    # 券種をDBのカラムまたはcomboの形式から判定
                    is_2t = (len(combo.split('-')) == 2)
                    
                    if is_2t:
                        result_str = res.get('nirentan_combo', '未確定')
                        payout = res.get('nirentan_payout', 0)
                    else:
                        result_str = res.get('sanrentan_combo', '未確定')
                        payout = res.get('sanrentan_payout', 0)
                    
                    if result_str != "未確定":
                        profit = payout - 100 if result_str == combo else -100
                        conn.execute("UPDATE history SET status='FINISHED', profit=? WHERE race_id=?", (profit, p['race_id']))
                        conn.commit()

                        today_str = p['date']
                        total_profit = conn.execute("SELECT SUM(profit) FROM history WHERE date=? AND status='FINISHED'", (today_str,)).fetchone()[0]
                        if total_profit is None: total_profit = 0

                        if result_str == combo:
                            msg = (
                                f"🎯 **{p['place']}{p['race_no']}R** 的中！({('2連単' if is_2t else '3連単')})\n"
                                f"買い目: {combo} ({p['odds']}倍)\n"
                                f"払戻: {payout:,}円 (収支: +{profit:,}円)\n"
                                f"📅 本日トータル: {total_profit:+,}円"
                            )
                            log(f"🎯 的中: {p['place']}{p['race_no']}R ({combo}) +{profit}円")
                            send_discord(msg)
                conn.close()
        except Exception: pass
        time.sleep(120)

def process_race(jcd, rno, today):
    with FINISHED_RACES_LOCK:
        if (jcd, rno) in FINISHED_RACES: return

    sess = get_session()
    place = PLACE_NAMES.get(jcd, "不明")
    
    try:
        raw, error = scrape_race_data(sess, jcd, rno, today)
    except:
        with STATS_LOCK: STATS["errors"] += 1
        return

    # ステータス判定の修正: OK以外はスルー
    if error != "OK" or not raw: return

    # 1. 予測実行 (会場フィルタリング含む)
    try:
        candidates, max_conf, is_target = predict_race(raw)
    except:
        with STATS_LOCK: STATS["errors"] += 1
        return

    if not is_target: return
    if not candidates: 
        # 戦略対象会場だが自信度不足
        return

    # 2. オッズ取得 (2T/3T 両方の可能性に対応)
    odds_2t, odds_3t = {}, {}
    has_2t = any(c['type'] == '2t' for c in candidates)
    has_3t = any(c['type'] == '3t' for c in candidates)
    
    try:
        if has_2t: odds_2t = get_odds_2t(sess, jcd, rno, today)
        if has_3t: odds_3t = get_odds_map(sess, jcd, rno, today)
    except Exception: pass

    # 3. EVフィルタリング (会場別・券種別の閾値を適用)
    # predict_boat.py 内の filter_and_sort_bets を使用
    try:
        final_bets, max_ev, current_thresh = filter_and_sort_bets(candidates, odds_2t, odds_3t, jcd)
    except: return

    if not final_bets:
        # 期待値不足で見送り
        with STATS_LOCK: STATS["vetted"] += 1
        return

    # 4. 時間管理
    deadline_str = raw.get('deadline_time')
    if deadline_str:
        try:
            now = datetime.datetime.now(JST)
            h, m = map(int, deadline_str.split(':'))
            deadline_dt = now.replace(hour=h, minute=m, second=0, microsecond=0)
            
            # 締切1分後まで判定を許容（スクレイピングのタイムラグ考慮）
            if now > (deadline_dt + datetime.timedelta(minutes=1)):
                with FINISHED_RACES_LOCK: FINISHED_RACES.add((jcd, rno))
                with STATS_LOCK: STATS["skipped"] += 1
                return

            # 20分以上前なら待機
            delta = deadline_dt - now
            if delta.total_seconds() > 1200:
                with STATS_LOCK: STATS["waiting"] += 1
                return
        except: pass

    # 5. 解説生成
    try:
        attach_reason(final_bets, raw, {})
    except Exception: pass

    # 6. DB保存 & 通知
    with STATS_LOCK: STATS["scanned"] += 1
    with DB_LOCK:
        conn = sqlite3.connect(DB_FILE)
        for p in final_bets:
            combo = p['combo']
            t_type = p['type']
            # race_idを重複防止のため券種まで含める
            race_id = f"{today}_{jcd}_{rno}_{combo}_{t_type}"
            
            if conn.execute("SELECT 1 FROM history WHERE race_id=?", (race_id,)).fetchone(): continue

            prob = float(p.get('prob', 0))
            reason = p.get('reason', '解説取得失敗')
            odds_val = p.get('odds', 0.0)
            ev_val = p.get('ev', 0.0)
            
            log(f"🔥 [HIT] {place}{rno}R ({t_type.upper()}) -> {combo} ({odds_val}倍 EV:{ev_val:.2f})")
            
            odds_url = f"https://www.boatrace.jp/owpc/pc/race/odds{'2t' if t_type=='2t' else '3t'}?rno={rno}&jcd={jcd:02d}&hd={today}"

            msg = (
                f"🔥 **{place}{rno}R** {t_type.upper()}激アツ\n"
                f"🎯 買い目: **{combo}**\n"
                f"📊 確率: **{prob}%** / オッズ: **{odds_val}倍**\n"
                f"💎 期待値: **{ev_val:.2f}**\n"
                f"📝 AI寸評: {reason}\n"
                f"🔗 [オッズ確認]({odds_url})"
            )
            
            conn.execute(
                "INSERT INTO history VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (race_id, today, place, rno, combo, 'PENDING', 0, odds_val, prob, ev_val, reason, t_type)
            )
            conn.commit()
            send_discord(msg)
            with STATS_LOCK: STATS["hits"] += 1
        conn.close()

def main():
    log(f"🚀 ハイブリッドAI Bot (ROI130% & 黄金律) 起動")
    
    try:
        load_model()
        log("✅ AIモデル(2T/3T) 読み込み完了")
    except Exception as e:
        error_log(f"FATAL: モデル読み込みエラー: {e}")
        sys.exit(1)

    init_db()
    stop_event = threading.Event()
    t = threading.Thread(target=report_worker, args=(stop_event,), daemon=True)
    t.start()
    
    start_time = time.time()
    MAX_RUNTIME = 18000 # 5時間
    
    while True:
        if time.time() - start_time > MAX_RUNTIME:
            log("🔄 5時間経過のため終了")
            break
        
        now = datetime.datetime.now(JST)
        if now.hour == 23 and now.minute >= 55: break
            
        today = now.strftime('%Y%m%d')
        
        with STATS_LOCK:
            STATS["scanned"] = 0; STATS["hits"] = 0; STATS["errors"] = 0
            STATS["skipped"] = 0; STATS["vetted"] = 0; STATS["waiting"] = 0

        log(f"🔍 直近レーススキャン中 ({today})...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
            for rno in range(1, 13):
                for jcd in range(1, 25):
                    ex.submit(process_race, jcd, rno, today)

        log(f"🏁 判定完了: 購入={STATS['hits']}, 見送り(EV不足)={STATS['vetted']}, 待機={STATS['waiting']}")
        # スキャン間隔を短縮（直前オッズの変化を逃さないため）
        time.sleep(60)

    stop_event.set()

if __name__ == "__main__":
    main()
