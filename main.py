import os
import datetime
import time
import sqlite3
import concurrent.futures
import threading
import sys
import requests as std_requests

from scraper import scrape_race_data, scrape_odds, scrape_result, get_session
from predict_boat import predict_race

DB_FILE = "race_data.db"
BET_AMOUNT = 1000 
PLACE_NAMES = {i: n for i, n in enumerate(["","桐生","戸田","江戸川","平和島","多摩川","浜名湖","蒲郡","常滑","津","三国","びわこ","住之江","尼崎","鳴門","丸亀","児島","宮島","徳山","下関","若松","芦屋","福岡","唐津","大村"])}
JST = datetime.timezone(datetime.timedelta(hours=9), 'JST')

# 文字化け防止・即時出力
sys.stdout.reconfigure(encoding='utf-8')

def log(msg):
    print(msg, flush=True)

def send_discord(content):
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if url: 
        try: std_requests.post(url, json={"content": content}, timeout=10)
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
                
                # 結果取得
                res = scrape_result(sess, jcd, p['race_no'], p['date'])
                if not res: continue

                hit = False
                payout = 0
                combo = p['predict_combo'] # 1-2-3 など
                
                # 3連単か2連単か判定
                result_str = "未確定"
                if str(combo).count("-") == 2: # 3連単
                    if res['sanrentan_combo']:
                        result_str = res['sanrentan_combo']
                        if res['sanrentan_combo'] == combo:
                            hit = True
                            payout = res['sanrentan_payout'] * (BET_AMOUNT/100)
                else: # 2連単
                    if res['nirentan_combo']:
                        result_str = res['nirentan_combo']
                        if res['nirentan_combo'] == combo:
                            hit = True
                            payout = res['nirentan_payout'] * (BET_AMOUNT/100)
                
                # 結果が出ていれば更新
                if result_str != "未確定":
                    profit = int(payout - BET_AMOUNT)
                    conn.execute("UPDATE history SET status='FINISHED', profit=? WHERE race_id=?", (profit, p['race_id']))
                    conn.commit()
                    
                    if hit:
                        send_discord(f"🎯 **{p['place']}{p['race_no']}R** 的中！ {combo} (払戻:{int(payout)}円)")
                        log(f"🎯 {p['place']}{p['race_no']}R 的中！ {combo} (+{profit}円)")
                    else:
                        # 外れもログに出す
                        log(f"💀 {p['place']}{p['race_no']}R ハズレ... 予想:{combo} 結果:{result_str}")

            conn.close()
        except Exception as e:
            log(f"Report Error: {e}")
        time.sleep(600)

def process_race(jcd, rno, today):
    sess = get_session()
    place = PLACE_NAMES[jcd]
    
    # データ取得
    try:
        raw, error = scrape_race_data(sess, jcd, rno, today)
    except Exception as e:
        log(f"❌ {place}{rno}R: プログラムエラー {e}")
        return

    # エラーがあれば必ず理由を表示
    if error:
        if error == "NO_DATA":
            pass # データなしは多すぎるのでスルー（必要なら log 出す）
        else:
            log(f"⚠️ {place}{rno}R: 取得失敗 ({error})")
        return

    # データチェック
    if not raw or raw.get('wr1', 0) == 0:
        log(f"⚠️ {place}{rno}R: データ欠損 (勝率0.0)")
        return
    
    # 成功ログ
    log(f"✅ {place}{rno}R 成功 [風:{raw['wind']}m] 1号艇(勝率:{raw['wr1']} モータ:{raw['mo1']})") 

    # 予測
    try:
        preds = predict_race(raw)
    except Exception as e:
        log(f"❌ {place}{rno}R: 予測エラー {e}")
        return

    if not preds:
        return

    conn = sqlite3.connect(DB_FILE)
    for p in preds:
        race_id = f"{today}_{jcd}_{rno}_{p['combo']}"
        exists = conn.execute("SELECT 1 FROM history WHERE race_id=?", (race_id,)).fetchone()
        
        if not exists:
            log(f"🔥 [HIT] {place}{rno}R 発見 -> {p['combo']}")
            conn.execute("INSERT INTO history VALUES (?,?,?,?,?,?,?)", (race_id, today, place, rno, p['combo'], 'PENDING', 0))
            conn.commit()
            send_discord(f"🔥 **{place}{rno}R** 推奨:[{p['type']}] {p['combo']} (実績期待値:{p['profit']}円)")
    conn.close()

def main():
    log("🚀 最強AI Bot (ログ全開・3連単対応版) 起動")
    init_db()
    threading.Thread(target=report_worker, daemon=True).start()
    
    while True:
        today = datetime.datetime.now(JST).strftime('%Y%m%d')
        log(f"⚡ Scan Start: {datetime.datetime.now(JST).strftime('%H:%M:%S')}")
        
        # ログが混ざらないよう並列数5で実行
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
            for jcd in range(1, 25):
                for rno in range(1, 13):
                    ex.submit(process_race, jcd, rno, today)
        
        log("💤 休憩中...")
        time.sleep(300)

if __name__ == "__main__":
    main()
