import os
import datetime
import time
import sqlite3
import concurrent.futures
import threading
import sys
import requests as std_requests

# 自作モジュール
from scraper import scrape_race_data, scrape_odds, scrape_result, get_session
from predict_boat import predict_race

DB_FILE = "race_data.db"
BET_AMOUNT = 1000 
PLACE_NAMES = {i: n for i, n in enumerate(["","桐生","戸田","江戸川","平和島","多摩川","浜名湖","蒲郡","常滑","津","三国","びわこ","住之江","尼崎","鳴門","丸亀","児島","宮島","徳山","下関","若松","芦屋","福岡","唐津","大村"])}
JST = datetime.timezone(datetime.timedelta(hours=9), 'JST')

sys.stdout.reconfigure(encoding='utf-8')

def log(msg):
    print(msg, flush=True)

def send_discord(content):
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if url: 
        try:
            std_requests.post(url, json={"content": content}, timeout=10)
        except: pass

def init_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("CREATE TABLE IF NOT EXISTS history (race_id TEXT PRIMARY KEY, date TEXT, place TEXT, race_no INTEGER, predict_combo TEXT, status TEXT, profit INTEGER)")
    conn.close()

def report_worker():
    while True:
        try:
            conn = sqlite3.connect(DB_FILE)
            conn.row_factory = sqlite3.Row
            pending = conn.execute("SELECT * FROM history WHERE status='PENDING'").fetchall()
            sess = get_session()
            for p in pending:
                try: jcd = int(p['race_id'].split('_')[1])
                except: continue
                
                res = scrape_result(sess, jcd, p['race_no'], p['date'])
                if res and res['nirentan_combo']:
                    hit = (p['predict_combo'] == res['nirentan_combo'])
                    payout = res['nirentan_payout'] * (BET_AMOUNT/100) if hit else 0
                    profit = int(payout - BET_AMOUNT)
                    conn.execute("UPDATE history SET status='FINISHED', profit=? WHERE race_id=?", (profit, p['race_id']))
                    conn.commit()
                    icon = "🎯" if hit else "💀"
                    send_discord(f"{icon} **{p['place']}{p['race_no']}R** 予想:{p['predict_combo']} 収支:{profit:+d}円")
            conn.close()
        except Exception as e:
            log(f"⚠️ Report Worker Error: {e}")
        time.sleep(600)

def process_race(jcd, rno, today):
    sess = get_session()
    place = PLACE_NAMES[jcd]
    
    try:
        raw = scrape_race_data(sess, jcd, rno, today)
    except Exception as e:
        log(f"❌ {place}{rno}R: エラー {e}")
        return

    if not raw:
        return
    
    if raw.get('wr1', 0) == 0:
        log(f"⚠️ {place}{rno}R: データ欠損 (勝率0.0)")
        return
    
    # ★★★ ここ修正：全データを整形して吐き出させる ★★★
    log(f"✅ {place}{rno}R [証明ログ] ----------------------------------")
    log(f"   風速: {raw.get('wind')}m | 締切: {raw.get('deadline_time')}")
    log(f"   1号艇: 勝率{raw['wr1']} / モーター{raw['mo1']} / ST{raw['st1']} / 展示{raw['ex1']}")
    log(f"   2号艇: 勝率{raw['wr2']} / モーター{raw['mo2']} / ST{raw['st2']} / 展示{raw['ex2']}")
    log(f"   3号艇: 勝率{raw['wr3']} / モーター{raw['mo3']} / ST{raw['st3']} / 展示{raw['ex3']}")
    log(f"   4号艇: 勝率{raw['wr4']} / モーター{raw['mo4']} / ST{raw['st4']} / 展示{raw['ex4']}")
    log(f"   5号艇: 勝率{raw['wr5']} / モーター{raw['mo5']} / ST{raw['st5']} / 展示{raw['ex5']}")
    log(f"   6号艇: 勝率{raw['wr6']} / モーター{raw['mo6']} / ST{raw['st6']} / 展示{raw['ex6']}")
    log(f"----------------------------------------------------------")

    try:
        preds = predict_race(raw)
    except: return

    if not preds: return

    conn = sqlite3.connect(DB_FILE)
    for p in preds:
        race_id = f"{today}_{jcd}_{rno}_{p['combo']}"
        exists = conn.execute("SELECT 1 FROM history WHERE race_id=?", (race_id,)).fetchone()
        
        if not exists:
            log(f"🔥 [HIT] {place}{rno}R -> {p['combo']}")
            conn.execute("INSERT INTO history VALUES (?,?,?,?,?,?,?)", (race_id, today, place, rno, p['combo'], 'PENDING', 0))
            conn.commit()
            send_discord(f"🔥 **{place}{rno}R** 推奨:[{p['type']}] {p['combo']} (実績期待値:{p['profit']}円)")
    conn.close()

def main():
    log("🚀 最強AI Bot (データ全開示証明版) 起動")
    init_db()
    threading.Thread(target=report_worker, daemon=True).start()
    
    while True:
        today = datetime.datetime.now(JST).strftime('%Y%m%d')
        log(f"⚡ Scan Start: {datetime.datetime.now(JST).strftime('%H:%M:%S')}")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
            for jcd in range(1, 25):
                for rno in range(1, 13):
                    ex.submit(process_race, jcd, rno, today)
        
        log("💤 スキャン完了。5分待機...")
        time.sleep(300)

if __name__ == "__main__":
    main()
