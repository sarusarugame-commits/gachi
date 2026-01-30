import os
import datetime
import time
import sqlite3
import concurrent.futures
import threading
import sys
import requests as std_requests
import json

# 自作モジュール
from scraper import scrape_race_data, get_session
# ★ predict_boat を読み込む
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
        resp = std_requests.post(url, json={"content": content}, timeout=10)
        if 200 <= resp.status_code < 300:
            log(f"✅ Discord送信成功: {resp.status_code}")
        else:
            log(f"💀 Discord送信失敗: Code {resp.status_code}")
    except Exception as e:
        log(f"💀 Discord接続エラー: {e}")

def init_db():
    conn = sqlite3.connect(DB_FILE)
    # 履歴テーブル作成
    conn.execute("CREATE TABLE IF NOT EXISTS history (race_id TEXT PRIMARY KEY, date TEXT, place TEXT, race_no INTEGER, predict_combo TEXT, status TEXT, profit INTEGER)")
    conn.close()
    log("💾 DB接続完了")

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

                hit = False
                payout = 0
                combo = p['predict_combo']
                result_str = "未確定"
                
                # 3連単の結果判定
                if str(combo).count("-") == 2:
                    if res.get('sanrentan_combo'):
                        result_str = res['sanrentan_combo']
                        if res['sanrentan_combo'] == combo:
                            hit = True
                            payout = res.get('sanrentan_payout', 0)
                
                if result_str != "未確定":
                    # 1点あたり100円計算で収支確定
                    profit = int(payout - 100)
                    conn.execute("UPDATE history SET status='FINISHED', profit=? WHERE race_id=?", (profit, p['race_id']))
                    conn.commit()
                    
                    if hit:
                        msg = f"🎯 **{p['place']}{p['race_no']}R** 的中！！\n買い目: **{combo}**\n払戻: {int(payout):,}円"
                        log(f"🎯 {p['place']}{p['race_no']}R 的中！ {combo} (+{profit}円)")
                        send_discord(msg)
                    else:
                        # ★ここを追加：ハズレ時もDiscordに通知
                        msg = f"💀 **{p['place']}{p['race_no']}R** ハズレ...\n予想: **{combo}**\n結果: {result_str}"
                        log(f"💀 {p['place']}{p['race_no']}R ハズレ... 予想:{combo} 結果:{result_str}")
                        send_discord(msg)

            conn.close()
        except Exception as e:
            log(f"Report Error: {e}")
        
        for _ in range(10):
            if stop_event.is_set(): break
            time.sleep(60)

def process_race(jcd, rno, today):
    sess = get_session()
    place = PLACE_NAMES.get(jcd, "不明")
    
    # 1. scraper.py を使ってデータ取得
    try:
        raw, error = scrape_race_data(sess, jcd, rno, today)
    except Exception as e:
        return

    if error: return
    # データ不備チェック
    if not raw or raw.get('wr1', 0) == 0: return

    # 2. predict_boat.py で予測 & 戦略判定
    try:
        preds = predict_race(raw)
    except Exception as e:
        return

    if not preds: return

    # 3. 激アツ買い目があればDB保存 & Discord通知
    conn = sqlite3.connect(DB_FILE)
    messages = []
    
    for p in preds:
        combo = p['combo']
        race_id = f"{today}_{jcd}_{rno}_{combo}"
        
        exists = conn.execute("SELECT 1 FROM history WHERE race_id=?", (race_id,)).fetchone()
        
        if not exists:
            prob = p.get('prob', 0)
            reason = p.get('reason', '')
            
            log(f"🔥 [HIT] {place}{rno}R -> {combo} (自信度:{prob}%)")
            
            # DB保存
            conn.execute("INSERT INTO history VALUES (?,?,?,?,?,?,?)", (race_id, today, place, rno, combo, 'PENDING', 0))
            
            messages.append(f"🎯 **{combo}** (自信度{prob}%)")

    if messages:
        conn.commit()
        odds_url = f"https://www.boatrace.jp/owpc/pc/race/odds3t?rno={rno}&jcd={jcd:02d}&hd={today}"
        
        msg = (
            f"🔥 **{place}{rno}R** 勝負レース！\n"
            f"{'\n'.join(messages)}\n"
            f"📝 {reason}\n"
            f"🔗 [オッズ確認]({odds_url})"
        )
        send_discord(msg)
            
    conn.close()

def main():
    log("🚀 最強AI Bot (Main + Predict Module) 起動")
    init_db()
    
    stop_event = threading.Event()
    t = threading.Thread(target=report_worker, args=(stop_event,), daemon=True)
    t.start()
    
    start_time = time.time()
    MAX_RUNTIME = 5.8 * 3600

    while True:
        now = datetime.datetime.now(JST)
        if now.hour == 23 and now.minute >= 55:
            log("🌙 ミッドナイト終了")
            break
        if time.time() - start_time > MAX_RUNTIME:
            log("🔄 タイムアウト")
            break

        today = now.strftime('%Y%m%d')
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
            for jcd in range(1, 25):
                for rno in range(1, 13):
                    ex.submit(process_race, jcd, rno, today)
        
        # 5分待機
        time.sleep(300)

    stop_event.set()
    log("👋 Bot停止")

if __name__ == "__main__":
    main()
