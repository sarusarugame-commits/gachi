import os
import datetime
import time
import sqlite3
import concurrent.futures
import threading
import sys
import requests as std_requests
import json

from scraper import scrape_race_data, get_session, get_odds_map, get_odds_2t, scrape_result
from predict_boat import predict_race, filter_and_sort_bets, attach_reason, load_model

DB_FILE = "race_data.db"
PLACE_NAMES = {i: n for i, n in enumerate(["","桐生","戸田","江戸川","平和島","多摩川","浜名湖","蒲郡","常滑","津","三国","びわこ","住之江","尼崎","鳴門","丸亀","児島","宮島","徳山","下関","若松","芦屋","福岡","唐津","大村"])}
JST = datetime.timezone(datetime.timedelta(hours=9), 'JST')

sys.stdout.reconfigure(encoding='utf-8')

DB_LOCK = threading.Lock()
STATS = {"scanned": 0, "hits": 0, "errors": 0, "skipped": 0, "waiting": 0}
STATS_LOCK = threading.Lock()
FINISHED_RACES = set()
FINISHED_RACES_LOCK = threading.Lock()

def log(msg):
    print(f"[{datetime.datetime.now(JST).strftime('%H:%M:%S')}] {msg}", flush=True)

def error_log(msg):
    print(f"[{datetime.datetime.now(JST).strftime('%H:%M:%S')}] ❌ {msg}", file=sys.stderr, flush=True)

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
    log("ℹ️ レポート監視スレッド起動")
    while not stop_event.is_set():
        try:
            with DB_LOCK:
                conn = sqlite3.connect(DB_FILE); conn.row_factory = sqlite3.Row
                pending = conn.execute("SELECT * FROM history WHERE status='PENDING'").fetchall()
                sess = get_session()
                
                for p in pending:
                    try: jcd = int(p['race_id'].split('_')[1])
                    except: continue
                    
                    res = scrape_result(sess, jcd, p['race_no'], p['date'])
                    if not res: continue

                    combo = p['predict_combo']
                    is_2t = len(combo.split('-')) == 2
                    res_c = res.get('nirentan_combo') if is_2t else res.get('sanrentan_combo')
                    payout = res.get('nirentan_payout', 0) if is_2t else res.get('sanrentan_payout', 0)
                    
                    if res_c and res_c != "未確定":
                        profit = payout - 100 if res_c == combo else -100
                        conn.execute("UPDATE history SET status='FINISHED', profit=? WHERE race_id=?", (profit, p['race_id']))
                        conn.commit()

                        today_str = p['date']
                        total_profit = conn.execute("SELECT SUM(profit) FROM history WHERE date=? AND status='FINISHED'", (today_str,)).fetchone()[0]
                        if total_profit is None: total_profit = 0

                        if res_c == combo:
                            msg = f"🎯 **{p['place']}{p['race_no']}R** 的中！\n買い目: {combo}\n払戻: {payout:,}円\n収支: +{profit:,}円\n📅 本日計: {total_profit:+,}円"
                            send_discord(msg)
                            log(f"🎯 的中: {p['place']}{p['race_no']}R ({combo}) +{profit}円")
                conn.close()
        except: pass
        for _ in range(10):
            if stop_event.is_set(): break
            time.sleep(60)

def process_race(jcd, rno, today):
    with FINISHED_RACES_LOCK:
        if (jcd, rno) in FINISHED_RACES: return

    sess = get_session(); place = PLACE_NAMES.get(jcd, "不明")
    
    # 1. データ取得
    try:
        raw, error = scrape_race_data(sess, jcd, rno, today)
    except: return

    # "OK"ステータスを正常として受け入れ、それ以外はスルー
    if error != "OK" or not raw: return

    # 2. 予測実行 (戦略対象外ならログを出さず終了)
    try:
        candidates, max_conf, is_target = predict_race(raw)
    except: return

    if not is_target: return

    # 3. 判定 (自信度チェック)
    if not candidates:
        log(f"👀 {place}{rno}R 判定: 見送り (自信度不足 {max_conf:.1%})")
        return

    # 4. オッズ取得
    o_2t, o_3t = {}, {}
    has_2t = any(c['type'] == '2t' for c in candidates)
    has_3t = any(c['type'] == '3t' for c in candidates)
    try:
        if has_2t: o_2t = get_odds_2t(sess, jcd, rno, today)
        if has_3t: o_3t = get_odds_map(sess, jcd, rno, today)
    except: pass

    # 5. EVフィルタ
    final_bets, max_ev, thresh = filter_and_sort_bets(candidates, o_2t, o_3t, jcd)
    
    if not final_bets:
        log(f"👀 {place}{rno}R 判定: 見送り (期待値不足 MaxEV:{max_ev:.2f} / 基準{thresh})")
        return

    # 6. 時間判定
    deadline_str = raw.get('deadline_time'); now = datetime.datetime.now(JST)
    if deadline_str:
        try:
            h, m = map(int, deadline_str.split(':'))
            deadline_dt = now.replace(hour=h, minute=m, second=0, microsecond=0)
            if now > (deadline_dt + datetime.timedelta(minutes=1)):
                with FINISHED_RACES_LOCK: FINISHED_RACES.add((jcd, rno))
                return
            delta = deadline_dt - now; minutes_left = delta.total_seconds() / 60
            if minutes_left > 20:
                with STATS_LOCK: STATS["waiting"] += 1; return
        except: pass

    # 7. 購入実行・解説
    attach_reason(final_bets, raw)
    with STATS_LOCK: STATS["scanned"] += 1
    with DB_LOCK:
        conn = sqlite3.connect(DB_FILE)
        for p in final_bets:
            combo = p['combo']; race_id = f"{today}_{jcd}_{rno}_{combo}_{p['type']}"
            if conn.execute("SELECT 1 FROM history WHERE race_id=?", (race_id,)).fetchone(): continue
            
            m_str = p['type'].upper()
            log(f"🔥 [{m_str} BUY] {place}{rno}R -> {combo} (EV:{p['ev']:.2f})")
            send_discord(f"🔥 **{place}{rno}R** {m_str}勝負！\n🎯 {combo} (EV:{p['ev']:.2f} / {p['odds']}倍)\n📝 {p.get('reason','')}")
            conn.execute("INSERT INTO history VALUES (?,?,?,?,?,?,?)", (race_id, today, place, rno, combo, 'PENDING', 0))
            conn.commit()
            with STATS_LOCK: STATS["hits"] += 1
        conn.close()

def main():
    log("🚀 ハイブリッド独立判定Bot (5時間リミット仕様) 起動")
    try: load_model(); log("✅ AIモデル読み込み完了")
    except Exception as e: error_log(f"FATAL: {e}"); sys.exit(1)

    init_db(); stop_event = threading.Event()
    threading.Thread(target=report_worker, args=(stop_event,), daemon=True).start()
    
    start_time = time.time(); MAX_RUNTIME = 18000 # 5時間
    while time.time() - start_time < MAX_RUNTIME:
        now = datetime.datetime.now(JST)
        if now.hour == 23 and now.minute >= 55: break
        today = now.strftime('%Y%m%d')
        
        with STATS_LOCK:
            STATS["scanned"] = 0; STATS["hits"] = 0; STATS["waiting"] = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
            futures = [ex.submit(process_race, jcd, rno, today) for rno in range(1, 13) for jcd in range(1, 25)]
            concurrent.futures.wait(futures)

        log(f"🏁 判定完了: 購入={STATS['hits']}R, 待機中={STATS['waiting']}R"); time.sleep(180)
    stop_event.set()

if __name__ == "__main__": main()
