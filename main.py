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

# ==========================================
# ⚙️ 設定
# ==========================================
DB_FILE = "race_data.db"
BET_AMOUNT = 1000 # 金額はここで調整
PLACE_NAMES = {i: n for i, n in enumerate(["","桐生","戸田","江戸川","平和島","多摩川","浜名湖","蒲郡","常滑","津","三国","びわこ","住之江","尼崎","鳴門","丸亀","児島","宮島","徳山","下関","若松","芦屋","福岡","唐津","大村"])}
JST = datetime.timezone(datetime.timedelta(hours=9), 'JST')

def send_discord(content):
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if url: requests.post(url, json={"content": content}, timeout=10)

# ==========================================
# 🗄️ データベース初期化
# ==========================================
def init_db():
    conn = sqlite3.connect(DB_FILE)
    # レースごとの履歴テーブル
    conn.execute("""
        CREATE TABLE IF NOT EXISTS history (
            race_id TEXT PRIMARY KEY, 
            date TEXT, 
            place TEXT, 
            race_no INTEGER, 
            predict_combo TEXT, 
            status TEXT, 
            profit INTEGER
        )
    """)
    # 【追加】その日の合計収支を保存するテーブル
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_summary (
            date TEXT PRIMARY KEY, 
            total_profit INTEGER
        )
    """)
    conn.close()

# ==========================================
# 📊 結果報告 & 日計DB更新スレッド
# ==========================================
def report_worker():
    """結果を回収し、日計をDBに書き込むスレッド"""
    while True:
        try:
            conn = sqlite3.connect(DB_FILE)
            conn.row_factory = sqlite3.Row
            pending = conn.execute("SELECT * FROM history WHERE status='PENDING'").fetchall()
            
            if pending:
                sess = get_session()
                for p in pending:
                    # IDからjcd（会場コード）を抽出
                    try:
                        jcd = int(p['race_id'].split('_')[1])
                    except: continue

                    res = scrape_result(sess, jcd, p['race_no'], p['date'])
                    
                    if res and res['nirentan_combo']:
                        hit = (p['predict_combo'] == res['nirentan_combo'])
                        payout = res['nirentan_payout'] * (BET_AMOUNT/100) if hit else 0
                        profit = int(payout - BET_AMOUNT)
                        
                        # 1. 個別レースのステータス更新
                        conn.execute("UPDATE history SET status='FINISHED', profit=? WHERE race_id=?", (profit, p['race_id']))
                        conn.commit()
                        
                        # 2. その日の合計収支を再計算
                        today_str = p['date']
                        c = conn.cursor()
                        c.execute("SELECT sum(profit) FROM history WHERE date=? AND status='FINISHED'", (today_str,))
                        daily_total = c.fetchone()[0] or 0
                        
                        # 3. 日計をDBに書き込み（INSERT or REPLACE）
                        conn.execute("INSERT OR REPLACE INTO daily_summary (date, total_profit) VALUES (?, ?)", (today_str, daily_total))
                        conn.commit()
                        
                        # Discord通知
                        icon = "🎯" if hit else "💀"
                        send_discord(
                            f"{icon} **{p['place']}{p['race_no']}R** 予想:{p['predict_combo']}\n"
                            f"💰 レース収支: {profit:+d}円\n"
                            f"📈 本日累計(DB記録): {daily_total:+d}円"
                        )
            
            conn.close()
        except Exception as e:
            print(f"Report Error: {e}")
        
        time.sleep(600) # 10分おきにチェック

# ==========================================
# 🚤 レース処理
# ==========================================
def process_race(jcd, rno, today):
    try:
        sess = get_session()
        raw = scrape_race_data(sess, jcd, rno, today)
        if not raw: return
        
        # 予測実行
        preds = predict_race(raw)
        if not preds: return

        conn = sqlite3.connect(DB_FILE)
        for p in preds:
            # IDをユニークにする（日付_会場_レース_買い目）
            race_id = f"{today}_{jcd}_{rno}_{p['combo']}"
            exists = conn.execute("SELECT 1 FROM history WHERE race_id=?", (race_id,)).fetchone()
            
            if not exists:
                conn.execute(
                    "INSERT INTO history (race_id, date, place, race_no, predict_combo, status, profit) VALUES (?,?,?,?,?,?,?)", 
                    (race_id, today, PLACE_NAMES[jcd], rno, p['combo'], 'PENDING', 0)
                )
                conn.commit()
                send_discord(f"🔥 **{PLACE_NAMES[jcd]}{rno}R** 推奨:[{p['type']}] {p['combo']} (実績期待値:{p['profit']}円)")
        conn.close()
    except Exception as e:
        print(f"Process Error {jcd}#{rno}: {e}")

# ==========================================
# 🚀 メインループ
# ==========================================
def main():
    init_db()
    # 結果監視スレッド起動
    threading.Thread(target=report_worker, daemon=True).start()
    
    print("🚀 最強AI Bot (日計DB保存版) 起動...")
    
    while True:
        now = datetime.datetime.now(JST)
        # 夜間はスリープ（23:30〜08:00など）
        if now.hour == 23 and now.minute > 30:
            time.sleep(30000) # 約8時間
            continue
            
        today = now.strftime('%Y%m%d')
        print(f"⚡ Scan start: {now.strftime('%H:%M:%S')}")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            for jcd in range(1, 25):
                for rno in range(1, 13):
                    ex.submit(process_race, jcd, rno, today)
        
        time.sleep(300) # 5分間隔で巡回

if __name__ == "__main__":
    main()
