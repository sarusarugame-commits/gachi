import os
import datetime
import time
import sqlite3
import concurrent.futures
import threading
import sys
import requests as std_requests

# scraper.py と predict_boat.py はそのままでOK
from scraper import scrape_race_data, get_session, get_odds_map, get_odds_2t, scrape_result
from predict_boat import predict_race, attach_reason, load_models, filter_and_sort_bets

DB_FILE = "race_data.db"
PLACE_NAMES = {i: n for i, n in enumerate(["","桐生","戸田","江戸川","平和島","多摩川","浜名湖","蒲郡","常滑","津","三国","びわこ","住之江","尼崎","鳴門","丸亀","児島","宮島","徳山","下関","若松","芦屋","福岡","唐津","大村"])}
JST = datetime.timezone(datetime.timedelta(hours=9), 'JST')

# 日本語出力設定
sys.stdout.reconfigure(encoding='utf-8')

DB_LOCK = threading.Lock()
STATS = {"scanned": 0, "hits": 0, "errors": 0, "skipped": 0, "waiting": 0, "passed": 0}
STATS_LOCK = threading.Lock()

FINISHED_RACES = set()
FINISHED_RACES_LOCK = threading.Lock()
MISSING_RACES = set()
MISSING_RACES_LOCK = threading.Lock()

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
    conn.execute("CREATE TABLE IF NOT EXISTS history (race_id TEXT PRIMARY KEY, date TEXT, place TEXT, race_no INTEGER, predict_combo TEXT, status TEXT, profit INTEGER)")
    conn.close()

def report_worker(stop_event):
    log("ℹ️ レポート監視スレッド起動")
    while not stop_event.is_set():
        try:
            with DB_LOCK:
                conn = sqlite3.connect(DB_FILE)
                conn.row_factory = sqlite3.Row
                pending = conn.execute("SELECT * FROM history WHERE status='PENDING'").fetchall()
                sess = get_session()
                
                for p in pending:
                    try: jcd = int(p['race_id'].split('_')[1])
                    except: continue
                    
                    res = scrape_result(sess, jcd, p['race_no'], p['date'])
                    if not res: continue

                    bet_combo = p['predict_combo']
                    is_2t = len(bet_combo.split('-')) == 2
                    
                    if is_2t:
                        result_str = res.get('combo_2t', '未確定')
                        payout = res.get('payout_2t', 0)
                    else:
                        result_str = res.get('combo_3t', '未確定')
                        payout = res.get('payout_3t', 0)
                    
                    if result_str != "未確定":
                        if result_str == bet_combo:
                            profit = payout - 100
                            res_emoji = "🎯"
                        else:
                            profit = -100
                            res_emoji = "💀"
                        
                        conn.execute("UPDATE history SET status='FINISHED', profit=? WHERE race_id=?", (profit, p['race_id']))
                        conn.commit()

                        today_str = p['date']
                        total_profit = conn.execute("SELECT SUM(profit) FROM history WHERE date=? AND status='FINISHED'", (today_str,)).fetchone()[0]
                        if total_profit is None: total_profit = 0

                        msg = (
                            f"{res_emoji} **{p['place']}{p['race_no']}R** 結果確定\n"
                            f"予測: {bet_combo} -> 結果: {result_str}\n"
                            f"収支: {profit:+},円 (本日計: {total_profit:+,}円)"
                        )
                        log(f"{res_emoji} {p['place']}{p['race_no']}R 結果:{result_str} (予測:{bet_combo}) {profit:+}")
                        if profit > 0: send_discord(msg)
                conn.close()
        except Exception as e:
            pass
        
        for _ in range(10):
            if stop_event.is_set(): break
            time.sleep(6)

def process_race(jcd, rno, today):
    with FINISHED_RACES_LOCK:
        if (jcd, rno) in FINISHED_RACES: return
    with MISSING_RACES_LOCK:
        if (jcd, rno) in MISSING_RACES: return

    sess = get_session()
    place = PLACE_NAMES.get(jcd, "不明")
    
    # 1. データ取得
    try:
        raw, error = scrape_race_data(sess, jcd, rno, today)
    except Exception as e:
        with STATS_LOCK: STATS["errors"] += 1
        return

    # 開催なし
    if error == "NO_RACE":
        with MISSING_RACES_LOCK: MISSING_RACES.add((jcd, rno))
        return

    # ★修正箇所: error が "OK" 以外の場合のみエラー扱いにする
    if (error != "OK") or not raw:
        with STATS_LOCK: 
            STATS["errors"] += 1
            if STATS["errors"] <= 5:
                log(f"⚠️ {place}{rno}R データ取得失敗: {error}")
        return

    # 締切チェック
    deadline_str = raw.get('deadline_time')
    if deadline_str:
        try:
            now = datetime.datetime.now(JST)
            h, m = map(int, deadline_str.split(':'))
            deadline_dt = now.replace(hour=h, minute=m, second=0, microsecond=0)
            
            if now > deadline_dt:
                with FINISHED_RACES_LOCK: FINISHED_RACES.add((jcd, rno))
                with STATS_LOCK: STATS["skipped"] += 1
                log(f"⌛ {place}{rno}R 締切経過 (Skipped)")
                return

            delta = deadline_dt - now
            minutes_left = delta.total_seconds() / 60

            # 20分前ルール (デバッグ時はここを緩和してもよい)
            if minutes_left > 20:
                with STATS_LOCK: STATS["waiting"] += 1
                return
        except: pass

    # 2. 予測
    try:
        ret = predict_race(raw)
        if not ret or len(ret) != 3: return
        candidates, mode, max_conf = ret
    except Exception as e:
        with STATS_LOCK: STATS["errors"] += 1
        return

    if not candidates or not mode:
        with STATS_LOCK: 
            STATS["scanned"] += 1
            STATS["passed"] += 1
        return

    # 3. オッズ取得
    odds_map = {}
    try:
        if mode == '2t':
            odds_map = get_odds_2t(sess, jcd, rno, today)
        else:
            odds_map = get_odds_map(sess, jcd, rno, today)
    except: pass

    if not odds_map:
        with STATS_LOCK: STATS["errors"] += 1
        return

    # 4. EVフィルタ
    try:
        final_bets, max_ev, thresh = filter_and_sort_bets(candidates, odds_map, jcd, mode)
    except: return
    
    with STATS_LOCK: STATS["scanned"] += 1
    
    if not final_bets:
        with STATS_LOCK: STATS["passed"] += 1
        return

    # 5. 投票＆通知
    attach_reason(final_bets, raw, odds_map)
    with DB_LOCK:
        conn = sqlite3.connect(DB_FILE)
        for p in final_bets:
            combo = p['combo']
            race_id = f"{today}_{jcd}_{rno}_{combo}" 
            
            if conn.execute("SELECT 1 FROM history WHERE race_id=?", (race_id,)).fetchone(): continue
            
            log(f"🔥 [BUY {mode.upper()}] {place}{rno}R -> {combo} (EV:{p['ev']:.1f})")
            
            odds_url = f"https://www.boatrace.jp/owpc/pc/race/odds{mode}f?rno={rno}&jcd={jcd:02d}&hd={today}"
            
            msg = (
                f"🔥 **{place}{rno}R** 厳選{mode.upper()}勝負！\n"
                f"⏰ 締切: **{deadline_str}** (あと{minutes_left:.0f}分)\n"
                f"🎯 買い目: **{combo}**\n"
                f"💰 期待値: **{p['ev']:.2f}** (基準{thresh})\n"
                f"📊 確率: {p['prob']}% / オッズ: {p['odds']}倍\n"
                f"📝 {p.get('reason','')}\n"
                f"🔗 [オッズ確認]({odds_url})"
            )
            
            conn.execute("INSERT INTO history VALUES (?,?,?,?,?,?,?)", (race_id, today, place, rno, combo, 'PENDING', 0))
            conn.commit()
            send_discord(msg)
            with STATS_LOCK: STATS["hits"] += 1
        conn.close()

def main():
    log("🚀 ハイブリッドBot (2連単厳選 & ノイズ除去) 起動")
    
    try:
        load_models() 
        log("✅ モデル読み込み完了")
    except Exception as e:
        error_log(f"FATAL: モデル読み込みエラー: {e}")
        sys.exit(1)

    init_db()
    
    stop_event = threading.Event()
    t = threading.Thread(target=report_worker, args=(stop_event,), daemon=True)
    t.start()
    
    start_time = time.time()
    MAX_RUNTIME = 18000 
    
    while True:
        if time.time() - start_time > MAX_RUNTIME:
            log("🔄 稼働時間上限(5時間)に達したため停止します")
            break
        
        now = datetime.datetime.now(JST)
        if now.hour == 23 and now.minute >= 55:
            log("🌙 ミッドナイト終了")
            break
            
        today = now.strftime('%Y%m%d')
        
        with STATS_LOCK:
            STATS["scanned"] = 0
            STATS["hits"] = 0
            STATS["errors"] = 0
            STATS["skipped"] = 0
            STATS["waiting"] = 0
            STATS["passed"] = 0

        log(f"🔍 直前レースのスキャン中 ({today})...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
            futures = []
            for rno in range(1, 13):
                for jcd in range(1, 25):
                    futures.append(ex.submit(process_race, jcd, rno, today))
            concurrent.futures.wait(futures)

        log(f"🏁 判定完了: 対象={STATS['scanned']}R -> 見送={STATS['passed']}R, 購入={STATS['hits']}R "
            f"(待機={STATS['waiting']}R, 期限切={STATS['skipped']}R, エラー={STATS['errors']}R)")
        
        log("💤 180秒待機...")
        time.sleep(180)

    stop_event.set()

if __name__ == "__main__":
    main()
