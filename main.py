import os
import datetime
import time
import sqlite3
import concurrent.futures
import threading
import sys
import requests as std_requests
import pandas as pd

# 自作モジュール (scrape_race_data だけで全て取ってくるように変更しました)
from scraper import scrape_race_data, get_session
from predict_boat import predict_race

DB_FILE = "race_data.db"
PLACE_NAMES = {i: n for i, n in enumerate(["","桐生","戸田","江戸川","平和島","多摩川","浜名湖","蒲郡","常滑","津","三国","びわこ","住之江","尼崎","鳴門","丸亀","児島","宮島","徳山","下関","若松","芦屋","福岡","唐津","大村"])}
JST = datetime.timezone(datetime.timedelta(hours=9), 'JST')

sys.stdout.reconfigure(encoding='utf-8')

def log(msg):
    print(msg, flush=True)

def send_discord(content):
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if url: 
        try: std_requests.post(url, json={"content": content}, timeout=10)
        except: pass

def process_race(jcd, rno, today):
    sess = get_session()
    place = PLACE_NAMES[jcd]
    
    # 全42項目を取得
    try:
        raw, error = scrape_race_data(sess, jcd, rno, today)
    except Exception as e:
        log(f"❌ {place}{rno}R: エラー {e}")
        return

    if error:
        return # NO_DATA等は無視

    # 取得データの証明ログ (ご指定の並び順で表示)
    log(f"✅ {place}{rno}R 取得完了 ------------------------------")
    
    # ヘッダー順に値を整形して表示
    headers = [
        'date', 'jcd', 'rno', 'wind', 'res1', 'rank1', 'rank2', 'rank3',
        'tansho', 'nirentan', 'sanrentan', 'sanrenpuku', 'payout',
        'wr1', 'mo1', 'ex1', 'f1', 'st1',
        'wr2', 'mo2', 'ex2', 'f2', 'st2',
        'wr3', 'mo3', 'ex3', 'f3', 'st3',
        'wr4', 'mo4', 'ex4', 'f4', 'st4',
        'wr5', 'mo5', 'ex5', 'f5', 'st5',
        'wr6', 'mo6', 'ex6', 'f6', 'st6'
    ]
    
    # 簡易表示用のCSV行を作成
    values = [str(raw.get(k, '')) for k in headers]
    log(f"   DATA: {','.join(values)}")
    log("----------------------------------------------------------")

    # 予測実行 (予測ロジックに必要なキーは全て raw に含まれています)
    try:
        preds = predict_race(raw)
    except: return

    if not preds: return

    # Discord通知など
    for p in preds:
        log(f"🔥 [HIT] {place}{rno}R -> {p['combo']} (期待値:{p['profit']}円)")
        send_discord(f"🔥 **{place}{rno}R** 推奨 {p['combo']}")

def main():
    log("🚀 最強AI Bot (全項目完全取得版) 起動")
    
    while True:
        today = datetime.datetime.now(JST).strftime('%Y%m%d')
        log(f"⚡ Scan Start: {datetime.datetime.now(JST).strftime('%H:%M:%S')}")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
            for jcd in range(1, 25):
                for rno in range(1, 13):
                    ex.submit(process_race, jcd, rno, today)
        
        log("💤 休憩中...")
        time.sleep(300)

if __name__ == "__main__":
    main()
