import os
import datetime
import time
import sqlite3
import concurrent.futures
import threading
import sys
import requests as std_requests
import json
import pandas as pd

# 自作モジュール
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
    # ★本番仕様: DROP TABLEを削除し、データを永続化させる
    conn.execute("CREATE TABLE IF NOT EXISTS history (race_id TEXT PRIMARY KEY, date TEXT, place TEXT, race_no INTEGER, predict_combo TEXT, status TEXT, profit INTEGER)")
    conn.close()
    log("💾 DB接続完了（履歴保持モード）")

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
                
                if str(combo).count("-") == 2:
                    if res.get('sanrentan_combo'):
                        result_str = res['sanrentan_combo']
                        if res['sanrentan_combo'] == combo:
                            hit = True
                            payout = res.get('sanrentan_payout', 0) * 10
                else:
                    if res.get('nirentan_combo'):
                        result_str = res['nirentan_combo']
                        if res['nirentan_combo'] == combo:
                            hit = True
                            payout = res.get('nirentan_payout', 0) * 10
                
                if result_str != "未確定":
                    profit = int(payout - 1000)
                    conn.execute("UPDATE history SET status='FINISHED', profit=? WHERE race_id=?", (profit, p['race_id']))
                    conn.commit()
                    
                    if hit:
                        msg = f"🎯 **{p['place']}{p['race_no']}R** 的中！！\n買い目: **{combo}**\n払戻: {int(payout):,}円\n収支: +{profit:,}円"
                        log(f"🎯 {p['place']}{p['race_no']}R 的中！ {combo} (+{profit}円)")
                        send_discord(msg)
                    else:
                        log(f"💀 {p['place']}{p['race_no']}R ハズレ... 予想:{combo} 結果:{result_str}")
            conn.close()
        except Exception as e:
            log(f"Report Error: {e}")
        
        for _ in range(10):
            if stop_event.is_set(): break
            time.sleep(60)

def process_race(jcd, rno, today):
    sess = get_session()
    place = PLACE_NAMES[jcd]
    try:
        raw, error = scrape_race_data(sess, jcd, rno, today)
    except Exception as e:
        return

    if error: return
    if not raw or raw.get('wr1', 0) == 0: return

    log(f"✅ {place}{rno}R 取得完了 ------------------------------")
    log("----------------------------------------------------------")

    try: preds = predict_race(raw)
    except: return
    if not preds: return

    conn = sqlite3.connect(DB_FILE)
    for p in preds:
        combo = p['combo']
        race_id = f"{today}_{jcd}_{rno}_{combo}"
        
        # 重複チェック（既に通知済みならスキップ）
        exists = conn.execute("SELECT 1 FROM history WHERE race_id=?", (race_id,)).fetchone()
        
        if not exists:
            ptype = p.get('type', '不明')
            profit = p.get('profit', 0)
            prob = p.get('prob', 0)
            roi = p.get('roi', 0)
            reason = p.get('reason', 'AI解説なし')
            
            log(f"🔥 [HIT] {place}{rno}R -> {combo} (期待値:{profit}円/確率:{prob}%)")
            odds_url = f"https://www.boatrace.jp/owpc/pc/race/odds3t?rno={rno}&jcd={jcd:02d}&hd={today}"

            msg = (
                f"🔥 **{place}{rno}R** AI激熱予想\n"
                f"🎯 買い目: **{combo}** ({ptype})\n"
                f"💰 期待値: **+{profit}円**\n"
                f"📊 自信度: **{prob}%** (回収率:{roi}%)\n"
                f"📝 **AI解説**: {reason}\n"
                f"🔗 [オッズ確認・投票]({odds_url})"
            )
            
            conn.execute("INSERT INTO history VALUES (?,?,?,?,?,?,?)", (race_id, today, place, rno, combo, 'PENDING', 0))
            conn.commit()
            send_discord(msg)
            
    conn.close()

def main():
    log("🚀 最強AI Bot (本番運用モード) 起動")
    init_db()
    
    stop_event = threading.Event()
    t = threading.Thread(target=report_worker, args=(stop_event,), daemon=True)
    t.start()
    
    start_time = time.time()
    MAX_RUNTIME = 5.8 * 3600

    while True:
        now = datetime.datetime.now(JST)
        # ミッドナイト終了時刻
        if now.hour == 23 and now.minute >= 55:
            log(f"🌙 {now.strftime('%H:%M')} ミッドナイト終了。")
            break
        # GitHub Actionsのタイムアウト対策
        if time.time() - start_time > MAX_RUNTIME:
            log("🔄 稼働時間上限。")
            break

        today = now.strftime('%Y%m%d')
        log(f"⚡ Scan Start: {now.strftime('%H:%M:%S')}")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
            for jcd in range(1, 25):
                for rno in range(1, 13):
                    ex.submit(process_race, jcd, rno, today)
        
        log("💤 休憩中...")
        time.sleep(300)

    stop_event.set()
    log("👋 Bot停止")

if __name__ == "__main__":
    main()
