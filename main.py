import os
import datetime
import time
import sqlite3
import concurrent.futures
import threading
import sys
import requests as std_requests

from scraper import scrape_race_data, get_session, get_odds_map, get_odds_2t, scrape_result
from predict_boat import predict_race, attach_reason, load_models, filter_and_sort_bets

DB_FILE = "race_data.db"
PLACE_NAMES = {i: n for i, n in enumerate(["","桐生","戸田","江戸川","平和島","多摩川","浜名湖","蒲郡","常滑","津","三国","びわこ","住之江","尼崎","鳴門","丸亀","児島","宮島","徳山","下関","若松","芦屋","福岡","唐津","大村"])}
JST = datetime.timezone(datetime.timedelta(hours=9), 'JST')
sys.stdout.reconfigure(encoding='utf-8')

DB_LOCK = threading.Lock()
STATS = {"scanned": 0, "hits": 0, "errors": 0, "skipped": 0, "waiting": 0, "passed": 0}
STATS_LOCK = threading.Lock()
FINISHED_RACES = set()
FINISHED_RACES_LOCK = threading.Lock()
MISSING_RACES = set()
MISSING_RACES_LOCK = threading.Lock()

def log(msg): print(f"[{datetime.datetime.now(JST).strftime('%H:%M:%S')}] {msg}", flush=True)

def send_discord(content):
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not url: return
    try: std_requests.post(url, json={"content": content}, timeout=10)
    except: pass

def init_db():
    conn = sqlite3.connect(DB_FILE)
    # typeカラムを追加したいが、既存DBがある場合はマイグレーションが必要。
    # ここでは簡易的に、predict_comboに "2t:1-2" のようにプレフィックスをつけるか、
    # 新規作成前提とする。
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
                
                for p in pending:
                    try: jcd = int(p['race_id'].split('_')[1])
                    except: continue
                    
                    res = scrape_result(sess, jcd, p['race_no'], p['date'])
                    if not res: continue

                    # 予測内容: "1-2" (2t) or "1-2-3" (3t)
                    # DBには区別がないので、桁数で判断するか、保存時に工夫する
                    bet_combo = p['predict_combo']
                    is_2t = len(bet_combo.split('-')) == 2
                    
                    if is_2t:
                        result_str = res.get('combo_2t', '未確定')
                        payout = res.get('payout_2t', 0)
                    else:
                        result_str = res.get('combo_3t', '未確定')
                        payout = res.get('payout_3t', 0)
                    
                    if result_str != "未確定":
                        if result_str == bet_combo:
                            profit = payout - 100
                            res_emoji = "🎯"
                        else:
                            profit = -100
                            res_emoji = "💀"
                        
                        conn.execute("UPDATE history SET status='FINISHED', profit=? WHERE race_id=?", (profit, p['race_id']))
                        conn.commit()
                        
                        msg = f"{res_emoji} {p['place']}{p['race_no']}R 結果: {result_str} (予想:{bet_combo}) 収支:{profit:+}"
                        log(msg)
                        if profit > 0: send_discord(msg)
                conn.close()
        except: pass
        time.sleep(60)

def process_race(jcd, rno, today):
    with FINISHED_RACES_LOCK:
        if (jcd, rno) in FINISHED_RACES: return
    with MISSING_RACES_LOCK:
        if (jcd, rno) in MISSING_RACES: return

    sess = get_session()
    place = PLACE_NAMES.get(jcd, "不明")
    
    # 1. データ取得
    try: raw, error = scrape_race_data(sess, jcd, rno, today)
    except: return

    if error == "NO_RACE":
        with MISSING_RACES_LOCK: MISSING_RACES.add((jcd, rno))
        return
    if error or not raw: return

    # 締切チェック (省略)
    
    # 2. 予測 (モード判定)
    try:
        # candidates, mode ('2t' or '3t'), confidence
        candidates, mode, max_conf = predict_race(raw)
    except: return

    if not candidates:
        with STATS_LOCK: STATS["passed"] += 1
        return

    # 3. オッズ取得 (モードに合わせて使い分け)
    odds_map = {}
    if mode == '2t':
        odds_map = get_odds_2t(sess, jcd, rno, today)
    else:
        odds_map = get_odds_map(sess, jcd, rno, today)

    if not odds_map: return

    # 4. EVフィルタ
    final_bets, max_ev, thresh = filter_and_sort_bets(candidates, odds_map, jcd, mode)
    with STATS_LOCK: STATS["scanned"] += 1
    
    if not final_bets:
        with STATS_LOCK: STATS["passed"] += 1
        return

    # 5. 投票
    attach_reason(final_bets, raw, odds_map)
    with DB_LOCK:
        conn = sqlite3.connect(DB_FILE)
        for p in final_bets:
            combo = p['combo']
            race_id = f"{today}_{jcd}_{rno}_{combo}" # ID重複注意
            if conn.execute("SELECT 1 FROM history WHERE race_id=?", (race_id,)).fetchone(): continue
            
            log(f"🔥 [BUY {mode.upper()}] {place}{rno}R -> {combo} (EV:{p['ev']:.1f})")
            
            msg = (
                f"🔥 **{place}{rno}R** 厳選{mode.upper()}勝負！\n"
                f"🎯 買い目: **{combo}**\n"
                f"💰 期待値: **{p['ev']:.2f}** (基準{thresh})\n"
                f"📊 オッズ: {p['odds']}倍"
            )
            conn.execute("INSERT INTO history VALUES (?,?,?,?,?,?,?)", (race_id, today, place, rno, combo, 'PENDING', 0))
            conn.commit()
            send_discord(msg)
            with STATS_LOCK: STATS["hits"] += 1
        conn.close()

def main():
    log("🚀 ハイブリッドBot (2連単厳選 & ノイズ除去) 起動")
    load_models() # 初回ロード
    init_db()
    
    stop_event = threading.Event()
    t = threading.Thread(target=report_worker, args=(stop_event,), daemon=True)
    t.start()
    
    # メインループ (省略、既存のものを流用)
    # ... (前回のmain.pyと同じループ構造を使ってください)

if __name__ == "__main__":
    main()
