import os
import datetime
import time
import requests
import sqlite3
import traceback
from scraper import scrape_result

# ==========================================
# ⚙️ 設定エリア
# ==========================================
DB_FILE = "race_data.db"
BET_AMOUNT = 1000
REPORT_HOURS = [13, 18, 23]
PLACE_NAMES = {
    1: "桐生", 2: "戸田", 3: "江戸川", 4: "平和島", 5: "多摩川", 6: "浜名湖",
    7: "蒲郡", 8: "常滑", 9: "津", 10: "三国", 11: "びわこ", 12: "住之江",
    13: "尼崎", 14: "鳴門", 15: "丸亀", 16: "児島", 17: "宮島", 18: "徳山",
    19: "下関", 20: "若松", 21: "芦屋", 22: "福岡", 23: "唐津", 24: "大村"
}

t_delta = datetime.timedelta(hours=9)
JST = datetime.timezone(t_delta, 'JST')

def send_discord(content):
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not url: return
    try: requests.post(url, json={"content": content}, timeout=10)
    except: pass

def check_results():
    conn = sqlite3.connect(DB_FILE, timeout=30)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    c.execute("SELECT * FROM history WHERE status='PENDING'")
    pending_races = c.fetchall()
    
    print(f"🔎 [Report] 結果待ちレース: {len(pending_races)}件")
    
    updated_count = 0
    sess = requests.Session()
    
    for race in pending_races:
        try:
            parts = race['race_id'].split('_')
            date_str, jcd, rno = parts[0], int(parts[1]), int(parts[2])
            
            res = scrape_result(sess, jcd, rno, date_str)
            if res:
                is_win = 1 if race['predict_combo'] == res['combo'] else 0
                profit = (res['payout'] - BET_AMOUNT) if is_win else -BET_AMOUNT
                
                c.execute("""
                    UPDATE history 
                    SET result_combo=?, is_win=?, payout=?, profit=?, status='FINISHED' 
                    WHERE race_id=?
                """, (res['combo'], is_win, res['payout'], profit, race['race_id']))
                
                place = PLACE_NAMES.get(jcd, "会場")
                msg = (f"{'🎊 的中' if is_win else '💀 外れ'} {place}{rno}R\n"
                       f"予測:{race['predict_combo']} → 結果:{res['combo']}\n"
                       f"収支:{'+' if profit>0 else ''}{profit}円")
                send_discord(msg)
                print(f"📊 結果判明: {place}{rno}R ({'的中' if is_win else '外れ'})")
                updated_count += 1
                time.sleep(1)
        except Exception as e:
            print(f"⚠️ Error checking result {race['race_id']}: {e}")
            continue
            
    if updated_count > 0:
        conn.commit()
    conn.close()

def send_periodic_report(last_report_key):
    now = datetime.datetime.now(JST)
    today = now.strftime('%Y%m%d')
    current_key = f"{today}_{now.hour}"
    
    print(f"🕒 [Report] 現在時刻: {now.strftime('%H:%M')} (ReportKey: {current_key})")
    
    if now.hour not in REPORT_HOURS:
        print(f"ℹ️ [Report] 報告時間外です ({REPORT_HOURS})")
        return last_report_key
        
    if last_report_key == current_key:
        print(f"ℹ️ [Report] 既に報告済みです")
        return last_report_key
    
    if now.hour == 23 and now.minute < 5:
        print(f"⏳ [Report] 23時報告待機中 (23:05以降に送信)")
        return last_report_key

    conn = sqlite3.connect(DB_FILE, timeout=30)
    c = conn.cursor()
    c.execute("SELECT count(*), sum(is_win), sum(profit) FROM history WHERE date=? AND status='FINISHED'", (today,))
    cnt, wins, profit = c.fetchone()
    c.execute("SELECT count(*) FROM history WHERE date=? AND status='PENDING'", (today,))
    pending_cnt = c.fetchone()[0]
    conn.close()
    
    if (cnt or 0) == 0 and (pending_cnt or 0) == 0:
        print("ℹ️ [Report] 本日のデータなし")
        return last_report_key

    msg = (f"**📊 {now.hour}時の収支報告**\n"
           f"✅ 完了レース: {cnt or 0}R (的中: {wins or 0})\n"
           f"⏳ 結果待ち: {pending_cnt or 0}R\n"
           f"💵 本日収支: {'+' if (profit or 0)>0 else ''}{profit or 0}円")
    send_discord(msg)
    print(f"📢 定期報告送信完了: {now.hour}時")
    
    return current_key

def main():
    print("📋 [Report] 結果確認・報告Bot起動")
    last_report_key = ""
    
    while True:
        now = datetime.datetime.now(JST)
        
        if now.hour >= 23 and now.minute >= 30:
            print("🌙 業務終了")
            break
            
        check_results()
        last_report_key = send_periodic_report(last_report_key)
        
        print("⏳ [Report] 待機: 600秒")
        time.sleep(600)

if __name__ == "__main__":
    main()
