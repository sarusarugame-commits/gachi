import os
import datetime
import time
import sqlite3
import concurrent.futures
import threading
import sys
import requests as std_requests
import json

# attach_reason をインポート
from scraper import scrape_race_data, get_session, scrape_odds
from predict_boat import predict_race, attach_reason

DB_FILE = "race_data.db"
PLACE_NAMES = {i: n for i, n in enumerate(["","桐生","戸田","江戸川","平和島","多摩川","浜名湖","蒲郡","常滑","津","三国","びわこ","住之江","尼崎","鳴門","丸亀","児島","宮島","徳山","下関","若松","芦屋","福岡","唐津","大村"])}
JST = datetime.timezone(datetime.timedelta(hours=9), 'JST')

sys.stdout.reconfigure(encoding='utf-8')

# DB書き込み競合を防ぐロック
DB_LOCK = threading.Lock()

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
            with DB_LOCK:
                conn = sqlite3.connect(DB_FILE)
                conn.row_factory = sqlite3.Row
                pending = conn.execute("SELECT * FROM history WHERE status='PENDING'").fetchall()
                sess = get_session()
                
                # 処理中のデータがあれば更新
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
                        actual_pay = payout * 10
                        profit = int(actual_pay - 1000)
                        
                        conn.execute("UPDATE history SET status='FINISHED', profit=? WHERE race_id=?", (profit, p['race_id']))
                        conn.commit()

                        # 本日トータル
                        today_str = p['date']
                        total_profit = conn.execute("SELECT SUM(profit) FROM history WHERE date=? AND status='FINISHED'", (today_str,)).fetchone()[0]
                        if total_profit is None: total_profit = 0

                        if result_str == combo:
                            msg = (
                                f"🎯 **{p['place']}{p['race_no']}R** 的中！\n"
                                f"買い目: {combo}\n"
                                f"払戻: {actual_pay:,}円\n"
                                f"収支: +{profit:,}円\n"
                                f"📅 **本日トータル: {total_profit:+,}円**"
                            )
                            send_discord(msg)
                        else:
                            msg = (
                                f"💀 **{p['place']}{p['race_no']}R** ハズレ\n"
                                f"予想: {combo} (結果: {result_str})\n"
                                f"📅 **本日トータル: {total_profit:+,}円**"
                            )
                            send_discord(msg)
                conn.close()

        except Exception as e:
            # log(f"Report Worker Error: {e}")
            pass
        
        for _ in range(10):
            if stop_event.is_set(): break
            time.sleep(60)

def process_race(jcd, rno, today):
    sess = get_session()
    place = PLACE_NAMES.get(jcd, "不明")
    
    # 1. データ取得
    try: raw, error = scrape_race_data(sess, jcd, rno, today)
    except: return
    if error or not raw: return

    # 2. 予測実行（★ここではまだAPIを叩かない）
    try: preds = predict_race(raw)
    except: return
    if not preds: return

    # 3. DBをチェックして「新規の買い目」があるか確認
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
    
    # 新規がないなら終了（API節約）
    if not new_preds:
        return

    # 4. ★ここで初めてAPIを叩いて理由を生成（新規データのみ）
    # predsリスト全体に理由を付与する（new_predsはpredsの参照を持っているので反映される）
    try:
        attach_reason(preds, raw)
    except Exception as e:
        log(f"Reason Error: {e}")

    # 5. DB保存と通知
    with DB_LOCK:
        conn = sqlite3.connect(DB_FILE)
        for p in new_preds:
            combo = p['combo']
            race_id = f"{today}_{jcd}_{rno}_{combo}"
            
            # 再度チェック（念のため）
            if conn.execute("SELECT 1 FROM history WHERE race_id=?", (race_id,)).fetchone():
                continue

            prob = p['prob']
            reason = p.get('reason', '解説取得失敗')
            deadline = p.get('deadline', '不明')
            
            log(f"🔥 [HIT] {place}{rno}R -> {combo} (確率:{prob}%)")
            odds_url = f"https://www.boatrace.jp/owpc/pc/race/odds3t?rno={rno}&jcd={jcd:02d}&hd={today}"

            msg = (
                f"🔥 **{place}{rno}R** 激アツ予想\n"
                f"⏰ 締切: **{deadline}**\n"
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
    log("🚀 最強AI Bot (本番運用モード v3) 起動")
    
    # 最初にモデルを一度読み込む（スレッド競合対策）
    from predict_boat import load_model
    load_model()
    
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
        
        # 5スレッドで並列処理
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
            for jcd in range(1, 25):
                for rno in range(1, 13):
                    ex.submit(process_race, jcd, rno, today)
        
        # 5分待機
        time.sleep(300)

    stop_event.set()

if __name__ == "__main__":
    main()
