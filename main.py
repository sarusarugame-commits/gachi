import os
import datetime
import time
import requests
import sqlite3
import concurrent.futures
import threading
from collections import defaultdict

# 自作モジュール
from scraper import scrape_race_data, scrape_odds, scrape_result, get_session
from predict_boat import predict_race

DB_FILE = "race_data.db"
BET_AMOUNT = 1000 
PLACE_NAMES = {i: n for i, n in enumerate(["","桐生","戸田","江戸川","平和島","多摩川","浜名湖","蒲郡","常滑","津","三国","びわこ","住之江","尼崎","鳴門","丸亀","児島","宮島","徳山","下関","若松","芦屋","福岡","唐津","大村"])}
JST = datetime.timezone(datetime.timedelta(hours=9), 'JST')

def send_discord(content):
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if url: requests.post(url, json={"content": content}, timeout=10)

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
                res = scrape_result(sess, int(p['race_id'].split('_')[1]), p['race_no'], p['date'])
                if res and res['nirentan_combo']:
                    hit = (p['predict_combo'] == res['nirentan_combo'])
                    payout = res['nirentan_payout'] * (BET_AMOUNT/100) if hit else 0
                    profit = int(payout - BET_AMOUNT)
                    conn.execute("UPDATE history SET status='FINISHED', profit=? WHERE race_id=?", (profit, p['race_id']))
                    conn.commit()
                    icon = "🎯" if hit else "💀"
                    send_discord(f"{icon} **{p['place']}{p['race_no']}R** 予想:{p['predict_combo']} 収支:{profit:+d}円")
            conn.close()
        except: pass
        time.sleep(600)

def process_race(jcd, rno, today):
    sess = get_session()
    
    # ★デバッグ: 取得開始
    # print(f"[DEBUG] {PLACE_NAMES[jcd]}{rno}R: データ取得開始...")

    raw = scrape_race_data(sess, jcd, rno, today)
    
    if raw:
        # ★デバッグ: 取得成功（中身の確認）
        print(f"✅ [DEBUG] {PLACE_NAMES[jcd]}{rno}R: 取得成功 -> 1号艇勝率:{raw.get('wr1')} / モーター:{raw.get('mo1')}")
        
        # もし勝率が0.0なら怪しいので警告
        if raw.get('wr1') == 0.0:
             print(f"⚠️ [WARNING] {PLACE_NAMES[jcd]}{rno}R: 勝率が0.0です。スクレイピング失敗の可能性あり。")
    else:
        # ★デバッグ: 取得失敗
        # print(f"❌ [DEBUG] {PLACE_NAMES[jcd]}{rno}R: データ取得失敗（まだ情報がない、または構造変更）")
        return

    preds = predict_race(raw)
    if not preds: return

    conn = sqlite3.connect(DB_FILE)
    for p in preds:
        race_id = f"{today}_{jcd}_{rno}_{p['combo']}"
        exists = conn.execute("SELECT 1 FROM history WHERE race_id=?", (race_id,)).fetchone()
        if not exists:
            # 予測が出たらログにも出す
            print(f"🔥 [DEBUG] {PLACE_NAMES[jcd]}{rno}R: 予測あり！ {p['combo']}")
            
            conn.execute("INSERT INTO history VALUES (?,?,?,?,?,?,?)", (race_id, today, PLACE_NAMES[jcd], rno, p['combo'], 'PENDING', 0))
            conn.commit()
            send_discord(f"🔥 **{PLACE_NAMES[jcd]}{rno}R** 推奨:[{p['type']}] {p['combo']} (実績期待値:{p['profit']}円)")
    conn.close()

def main():
    init_db()
    threading.Thread(target=report_worker, daemon=True).start()
    print("🚀 最強AI Bot (デバッグモード) 起動中...")
    
    while True:
        today = datetime.datetime.now(JST).strftime('%Y%m%d')
        print(f"⚡ Scan start: {datetime.datetime.now(JST).strftime('%H:%M:%S')}")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            for jcd in range(1, 25):
                for rno in range(1, 13):
                    ex.submit(process_race, jcd, rno, today)
        time.sleep(300)

if __name__ == "__main__":
    main()
