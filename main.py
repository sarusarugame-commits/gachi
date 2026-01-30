import os
import datetime
import time
import sqlite3
import concurrent.futures
import threading
import sys
import requests as std_requests
import json

from scraper import scrape_race_data, get_session, scrape_odds
from predict_boat import predict_race

DB_FILE = "race_data.db"
PLACE_NAMES = {i: n for i, n in enumerate(["","桐生","戸田","江戸川","平和島","多摩川","浜名湖","蒲郡","常滑","津","三国","びわこ","住之江","尼崎","鳴門","丸亀","児島","宮島","徳山","下関","若松","芦屋","福岡","唐津","大村"])}
JST = datetime.timezone(datetime.timedelta(hours=9), 'JST')

sys.stdout.reconfigure(encoding='utf-8')

def log(msg):
    print(msg, flush=True)

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
    while not stop_event.is_set():
        try:
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
                    profit = -100 # 外れなら-100円
                    hit_mark = "💀"
                    
                    if result_str == combo:
                        profit = payout - 100
                        hit_mark = "🎯"
                        msg = f"{hit_mark} **{p['place']}{p['race_no']}R** 的中！\n買い目: {combo}\n払戻: {payout:,}円\n収支: +{profit:,}円"
                        send_discord(msg)
                    else:
                        msg = f"{hit_mark} **{p['place']}{p['race_no']}R** ハズレ\n予想: {combo}\n結果: {result_str}"
                        send_discord(msg)

                    conn.execute("UPDATE history SET status='FINISHED', profit=? WHERE race_id=?", (profit, p['race_id']))
                    conn.commit()
            conn.close()
        except: pass
        
        for _ in range(10):
            if stop_event.is_set(): break
            time.sleep(60)

def process_race(jcd, rno, today):
    sess = get_session()
    place = PLACE_NAMES.get(jcd, "不明")
    
    try: raw, error = scrape_race_data(sess, jcd, rno, today)
    except: return
    if error or not raw: return

    # オッズ取得 (今は空でもOK)
    odds = scrape_odds(sess, jcd, rno, today)

    try: preds = predict_race(raw, odds)
    except: return
    if not preds: return

    conn = sqlite3.connect(DB_FILE)
    for p in preds:
        combo = p['combo']
        race_id = f"{today}_{jcd}_{rno}_{combo}"
        exists = conn.execute("SELECT 1 FROM history WHERE race_id=?", (race_id,)).fetchone()
        
        if not exists:
            prob = p['prob']
            reason = p['reason']
            
            log(f"🔥 [HIT] {place}{rno}R -> {combo} (確率:{prob}%)")
            odds_url = f"https://www.boatrace.jp/owpc/pc/race/odds3t?rno={rno}&jcd={jcd:02d}&hd={today}"

            msg = (
                f"🔥 **{place}{rno}R** 激アツ予想\n"
                f"🎯 買い目: **{combo}**\n"
                f"📊 当選確率: **{prob}%**\n"
                f"📝 解説: {reason}\n"
                f"🔗 [オッズ確認]({odds_url})"
            )
            
            conn.execute("INSERT INTO history VALUES (?,?,?,?,?,?,?)", (race_id, today, place, rno, combo, 'PENDING', 0))
            conn.commit()
            send_discord(msg)
            
    conn.close()

def main():
    log("🚀 最強AI Bot 起動")
    init_db()
    stop_event = threading.Event()
    t = threading.Thread(target=report_worker, args=(stop_event,), daemon=True)
    t.start()
    
    start_time = time.time()
    while True:
        if time.time() - start_time > 21000: break # 5.8時間
        
        now = datetime.datetime.now(JST)
        today = now.strftime('%Y%m%d')
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
            for jcd in range(1, 25):
                for rno in range(1, 13):
                    ex.submit(process_race, jcd, rno, today)
        time.sleep(300)

    stop_event.set()

if __name__ == "__main__":
    main()
