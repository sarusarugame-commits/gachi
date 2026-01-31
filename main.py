import os
import datetime
import time
import sqlite3
import concurrent.futures
import threading
import sys
import requests as std_requests
import json

from scraper import scrape_race_data, get_session, scrape_odds, get_exact_odds
from predict_boat import predict_race, attach_reason, load_model

DB_FILE = "race_data.db"
PLACE_NAMES = {i: n for i, n in enumerate(["","桐生","戸田","江戸川","平和島","多摩川","浜名湖","蒲郡","常滑","津","三国","びわこ","住之江","尼崎","鳴門","丸亀","児島","宮島","徳山","下関","若松","芦屋","福岡","唐津","大村"])}
JST = datetime.timezone(datetime.timedelta(hours=9), 'JST')

sys.stdout.reconfigure(encoding='utf-8')

DB_LOCK = threading.Lock()
STATS = {"scanned": 0, "hits": 0, "errors": 0, "skipped": 0}
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
    conn.execute("CREATE TABLE IF NOT EXISTS history (race_id TEXT PRIMARY KEY, date TEXT, place TEXT, race_no INTEGER, predict_combo TEXT, status TEXT, profit INTEGER)")
    conn.close()

def report_worker(stop_event):
    log("ℹ️ レポート監視スレッド起動 (1点100円計算)")
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
                    
                    from scraper import scrape_result
                    res = scrape_result(sess, jcd, p['race_no'], p['date'])
                    if not res: continue

                    combo = p['predict_combo']
                    result_str = res.get('sanrentan_combo', '未確定')
                    payout = res.get('sanrentan_payout', 0)
                    
                    if result_str != "未確定":
                        if result_str == combo:
                            profit = payout - 100
                        else:
                            profit = -100
                        
                        conn.execute("UPDATE history SET status='FINISHED', profit=? WHERE race_id=?", (profit, p['race_id']))
                        conn.commit()

                        today_str = p['date']
                        total_profit = conn.execute("SELECT SUM(profit) FROM history WHERE date=? AND status='FINISHED'", (today_str,)).fetchone()[0]
                        if total_profit is None: total_profit = 0

                        if result_str == combo:
                            msg = (
                                f"🎯 **{p['place']}{p['race_no']}R** 的中！\n"
                                f"買い目: {combo}\n"
                                f"払戻: {payout:,}円\n"
                                f"収支: +{profit:,}円\n"
                                f"📅 **本日トータル: {total_profit:+,}円**"
                            )
                            log(f"🎯 的中: {p['place']}{p['race_no']}R ({combo}) +{profit}円")
                            send_discord(msg)
                        else:
                            msg = (
                                f"💀 **{p['place']}{p['race_no']}R** ハズレ\n"
                                f"予想: {combo} (結果: {result_str})\n"
                                f"📅 **本日トータル: {total_profit:+,}円**"
                            )
                            log(f"💀 ハズレ: {p['place']}{p['race_no']}R (結果:{result_str})")
                            send_discord(msg)
                conn.close()
        except Exception as e:
            pass
        
        for _ in range(10):
            if stop_event.is_set(): break
            time.sleep(60)

def process_race(jcd, rno, today):
    with FINISHED_RACES_LOCK:
        if (jcd, rno) in FINISHED_RACES:
            with STATS_LOCK: STATS["skipped"] += 1
            return

    sess = get_session()
    place = PLACE_NAMES.get(jcd, "不明")
    
    try:
        raw, error = scrape_race_data(sess, jcd, rno, today)
    except Exception as e:
        with STATS_LOCK: STATS["errors"] += 1
        return

    if error or not raw: return

    deadline_str = raw.get('deadline_time')
    if deadline_str:
        try:
            now = datetime.datetime.now(JST)
            h, m = map(int, deadline_str.split(':'))
            deadline_dt = now.replace(hour=h, minute=m, second=0, microsecond=0)
            
            if now > deadline_dt:
                with FINISHED_RACES_LOCK:
                    FINISHED_RACES.add((jcd, rno))
                with STATS_LOCK: STATS["skipped"] += 1
                return

            if deadline_dt > (now + datetime.timedelta(minutes=60)):
                with STATS_LOCK: STATS["skipped"] += 1
                return
        except: pass

    try:
        preds = predict_race(raw)
    except:
        with STATS_LOCK: STATS["errors"] += 1
        return

    with STATS_LOCK: STATS["scanned"] += 1
    if not preds: return

    new_preds = []
    with DB_LOCK:
        conn = sqlite3.connect(DB_FILE)
        for p in preds:
            combo = p['combo']
            race_id = f"{today}_{jcd}_{rno}_{combo}"
            exists = conn.execute("SELECT 1 FROM history WHERE race_id=?", (race_id,)).fetchone()
            if not exists:
                new_preds.append(p)
        conn.close()
    
    if not new_preds: return

    log(f"⚡ {place}{rno}R で {len(new_preds)}件の候補を検知！オッズ取得＆AI解説生成中...")
    
    # オッズ取得
    best_combo = new_preds[0]['combo']
    odds_val = None
    try:
        odds_val = get_exact_odds(sess, jcd, rno, today, best_combo)
        if odds_val:
            log(f"💰 {place}{rno}R: 現在オッズ {odds_val}倍 を取得")
    except Exception as e:
        log(f"⚠️ オッズ取得失敗: {e}")

    try:
        attach_reason(preds, raw, odds_val)
    except Exception as e:
        log(f"⚠️ 解説エラー: {e}")

    with DB_LOCK:
        conn = sqlite3.connect(DB_FILE)
        for p in new_preds:
            combo = p['combo']
            race_id = f"{today}_{jcd}_{rno}_{combo}"
            if conn.execute("SELECT 1 FROM history WHERE race_id=?", (race_id,)).fetchone(): continue

            prob = p['prob']
            reason = p.get('reason', '解説取得失敗')
            deadline = p.get('deadline', '不明')
            
            odds_log = f"({p.get('odds')}倍)" if p.get('odds') else ""
            log(f"🔥 [HIT] {place}{rno}R -> {combo} (確率:{prob}%) {odds_log}")
            
            odds_url = f"https://www.boatrace.jp/owpc/pc/race/odds3t?rno={rno}&jcd={jcd:02d}&hd={today}"

            msg = (
                f"🔥 **{place}{rno}R** 激アツ予想\n"
                f"⏰ 締切: **{deadline}**\n"
                f"🎯 買い目: **{combo}**\n"
                f"📊 当選確率: **{prob}%**\n"
                f"💰 現在オッズ: **{odds_val if odds_val else '不明'}倍**\n"
                f"📝 解説: {reason}\n"
                f"🔗 [オッズ確認]({odds_url})"
            )
            
            conn.execute("INSERT INTO history VALUES (?,?,?,?,?,?,?)", (race_id, today, place, rno, combo, 'PENDING', 0))
            conn.commit()
            send_discord(msg)
            with STATS_LOCK: STATS["hits"] += 1
        conn.close()

def main():
    log("🚀 最強AI Bot (本番運用モード v4.0) 起動 - オッズ分析機能搭載")
    
    try:
        load_model()
        log("✅ AIモデル読み込み完了")
    except Exception as e:
        error_log(f"FATAL: モデル読み込みエラー: {e}")
        sys.exit(1)

    init_db()
    stop_event = threading.Event()
    t = threading.Thread(target=report_worker, args=(stop_event,), daemon=True)
    t.start()
    
    start_time = time.time()
    MAX_RUNTIME = 21000 
    
    while True:
        if time.time() - start_time > MAX_RUNTIME:
            log("🔄 稼働時間上限により停止")
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

        log(f"🔍 直近のレースをスキャン中 ({today})...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
            futures = []
            for rno in range(1, 13):
                for jcd in range(1, 25):
                    futures.append(ex.submit(process_race, jcd, rno, today))
            concurrent.futures.wait(futures)

        log(f"🏁 スキャン完了: 有効チェック={STATS['scanned']}, 範囲外スキップ={STATS['skipped']}, HIT={STATS['hits']}")
        log("💤 待機中(300秒)...")
        time.sleep(300)

    stop_event.set()

if __name__ == "__main__":
    main()
