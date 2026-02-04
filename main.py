import os
import datetime
import time
import sqlite3
import concurrent.futures
import threading
import sys
import requests as std_requests
import json

# scraper.pyも更新が必要です
from scraper import scrape_race_data, get_session, get_odds_map
from predict_boat import predict_race, attach_reason, load_model, filter_and_sort_bets

DB_FILE = "race_data.db"
PLACE_NAMES = {i: n for i, n in enumerate(["","桐生","戸田","江戸川","平和島","多摩川","浜名湖","蒲郡","常滑","津","三国","びわこ","住之江","尼崎","鳴門","丸亀","児島","宮島","徳山","下関","若松","芦屋","福岡","唐津","大村"])}
JST = datetime.timezone(datetime.timedelta(hours=9), 'JST')

sys.stdout.reconfigure(encoding='utf-8')

DB_LOCK = threading.Lock()
# 統計情報の項目を網羅
STATS = {"scanned": 0, "hits": 0, "errors": 0, "skipped": 0, "waiting": 0, "passed": 0}
STATS_LOCK = threading.Lock()

FINISHED_RACES = set()
FINISHED_RACES_LOCK = threading.Lock()

# ★追加: 開催されていないレースを記憶するセット
MISSING_RACES = set()
MISSING_RACES_LOCK = threading.Lock()

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

                    combo = p['predict_combo']
                    result_str = res.get('sanrentan_combo', '未確定')
                    payout = res.get('sanrentan_payout', 0)
                    
                    if result_str != "未確定":
                        if result_str == combo:
                            profit = payout - 100
                        else:
                            profit = -100
                        
                        conn.execute("UPDATE history SET status='FINISHED', profit=? WHERE race_id=?", (profit, p['race_id']))
                        conn.commit()

                        today_str = p['date']
                        total_profit = conn.execute("SELECT SUM(profit) FROM history WHERE date=? AND status='FINISHED'", (today_str,)).fetchone()[0]
                        if total_profit is None: total_profit = 0

                        if result_str == combo:
                            msg = (
                                f"🎯 **{p['place']}{p['race_no']}R** 的中！\n"
                                f"買い目: {combo}\n"
                                f"払戻: {payout:,}円\n"
                                f"収支: +{profit:,}円\n"
                                f"📅 **本日トータル: {total_profit:+,}円**"
                            )
                            log(f"🎯 的中: {p['place']}{p['race_no']}R ({combo}) +{profit}円")
                            send_discord(msg)
                        else:
                            log(f"💀 ハズレ: {p['place']}{p['race_no']}R (結果:{result_str})")
                conn.close()
        except Exception as e:
            pass
        
        for _ in range(10):
            if stop_event.is_set(): break
            time.sleep(60)

def process_race(jcd, rno, today):
    # 既に終了したレースはスキップ
    with FINISHED_RACES_LOCK:
        if (jcd, rno) in FINISHED_RACES:
            return

    # ★追加: そもそも開催がない(NO_RACE)と判定済みの場合はスキップ
    with MISSING_RACES_LOCK:
        if (jcd, rno) in MISSING_RACES:
            return

    sess = get_session()
    place = PLACE_NAMES.get(jcd, "不明")
    
    # 1. データ取得
    try:
        # scraper.pyを更新し、errorとして詳細なコードを受け取る必要があります
        raw, error = scrape_race_data(sess, jcd, rno, today)
    except Exception as e:
        with STATS_LOCK: STATS["errors"] += 1
        log(f"⚠️ {place}{rno}R データ取得例外: {e}")
        return

    # ★重要: 「データなし(開催なし)」の場合は、無視リストに入れて次回からアクセスしない
    if error == "NO_RACE":
        with MISSING_RACES_LOCK:
            MISSING_RACES.add((jcd, rno))
        return

    # その他のエラー（通信失敗など）はカウントしてリトライ対象のままにする
    if error or not raw:
        with STATS_LOCK: STATS["errors"] += 1
        # 頻繁に出る場合はコメントアウト推奨
        # log(f"⚠️ {place}{rno}R データ取得失敗 ({error})") 
        return

    # ★時間管理ロジック★
    deadline_str = raw.get('deadline_time')
    if deadline_str:
        try:
            now = datetime.datetime.now(JST)
            h, m = map(int, deadline_str.split(':'))
            deadline_dt = now.replace(hour=h, minute=m, second=0, microsecond=0)
            
            # すでに締め切っていたら終了リストへ
            if now > deadline_dt:
                with FINISHED_RACES_LOCK: FINISHED_RACES.add((jcd, rno))
                with STATS_LOCK: STATS["skipped"] += 1
                log(f"⌛ {place}{rno}R 締切時刻経過 (Skipped)")
                return

            # 残り時間を計算 (分)
            delta = deadline_dt - now
            minutes_left = delta.total_seconds() / 60

            # 「締め切り20分前」になっていなければ、待機リストへ
            if minutes_left > 20:
                with STATS_LOCK: STATS["waiting"] += 1
                return
            
        except: pass

    # 2. 一次候補 (確率判定)
    try:
        # predict_race は (candidates, max_prob) を返す
        ret_predict = predict_race(raw)
        if isinstance(ret_predict, tuple):
            candidates, max_conf = ret_predict
        else:
            candidates = ret_predict
            max_conf = 0.0
    except Exception as e:
        with STATS_LOCK: STATS["errors"] += 1
        log(f"⚠️ {place}{rno}R 予測エラー: {e}")
        return

    if not candidates:
        with STATS_LOCK: 
            STATS["scanned"] += 1
            STATS["passed"] += 1
        log(f"👀 {place}{rno}R 見送り: 自信度不足 ({max_conf:.1%} < 15.0%)")
        return

    # 3. オッズ取得 (直前オッズ！)
    odds_map = {}
    try:
        odds_map = get_odds_map(sess, jcd, rno, today)
    except Exception as e:
        with STATS_LOCK: STATS["errors"] += 1
        log(f"⚠️ {place}{rno}R オッズ取得失敗: {e}")
        return

    if not odds_map:
        with STATS_LOCK: STATS["errors"] += 1
        return

    # 4. EVフィルタリング
    try:
        ret_filter = filter_and_sort_bets(candidates, odds_map, jcd)
        if isinstance(ret_filter, tuple):
            final_bets, max_ev, ev_thresh = ret_filter
        else:
            final_bets = ret_filter
            max_ev = 0.0
            ev_thresh = 0.0
    except Exception as e:
        with STATS_LOCK: STATS["errors"] += 1
        return
    
    with STATS_LOCK: STATS["scanned"] += 1
    
    if not final_bets:
        with STATS_LOCK: STATS["passed"] += 1
        log(f"👀 {place}{rno}R 見送り: 期待値不足 (Max EV:{max_ev:.2f} < {ev_thresh})")
        return

    log(f"⚡ {place}{rno}R (締切{minutes_left:.1f}分前) 勝負買い目あり！Groq解説生成中...")

    # 5. 解説付与
    try:
        attach_reason(final_bets, raw, odds_map)
    except Exception as e:
        log(f"⚠️ 解説生成エラー: {e}")

    # 6. 投票＆通知
    with DB_LOCK:
        conn = sqlite3.connect(DB_FILE)
        for p in final_bets:
            combo = p['combo']
            race_id = f"{today}_{jcd}_{rno}_{combo}"
            
            if conn.execute("SELECT 1 FROM history WHERE race_id=?", (race_id,)).fetchone(): continue

            prob = p['prob']
            odds_val = p.get('odds')
            ev_val = p.get('ev')
            reason = p.get('reason', '解説なし')
            
            log(f"🔥 [BUY] {place}{rno}R -> {combo} (EV:{ev_val:.2f})")
            
            odds_url = f"https://www.boatrace.jp/owpc/pc/race/odds3t?rno={rno}&jcd={jcd:02d}&hd={today}"

            msg = (
                f"🔥 **{place}{rno}R** 直前スナイプ！\n"
                f"⏰ 締切: **{deadline_str}** (あと{minutes_left:.0f}分)\n"
                f"🎯 買い目: **{combo}**\n"
                f"💰 期待値: **{ev_val:.2f}**\n"
                f"📊 確率: {prob}% / オッズ: {odds_val}倍\n"
                f"📝 解説: {reason}\n"
                f"🔗 [オッズ確認]({odds_url})"
            )
            
            conn.execute("INSERT INTO history VALUES (?,?,?,?,?,?,?)", (race_id, today, place, rno, combo, 'PENDING', 0))
            conn.commit()
            send_discord(msg)
            with STATS_LOCK: STATS["hits"] += 1
        conn.close()

def main():
    log("🚀 最強AI Bot (直前スナイプ & 5時間リミット版) 起動")
    
    try:
        load_model()
        log("✅ AIモデル読み込み完了")
    except Exception as e:
        error_log(f"FATAL: モデル読み込みエラー: {e}")
        sys.exit(1)

    init_db()
    stop_event = threading.Event()
    t = threading.Thread(target=report_worker, args=(stop_event,), daemon=True)
    t.start()
    
    start_time = time.time()
    MAX_RUNTIME = 18000
    
    while True:
        if time.time() - start_time > MAX_RUNTIME:
            log("🔄 稼働時間上限(5時間)に達したため停止します")
            break
        
        now = datetime.datetime.now(JST)
        if now.hour == 23 and now.minute >= 55:
            log("🌙 ミッドナイト終了")
            break
            
        today = now.strftime('%Y%m%d')
        
        with STATS_LOCK:
            STATS["scanned"] = 0
            STATS["hits"] = 0
            STATS["errors"] = 0
            STATS["skipped"] = 0
            STATS["waiting"] = 0
            STATS["passed"] = 0

        log(f"🔍 直前レースのスキャン中 ({today})...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
            futures = []
            for rno in range(1, 13):
                for jcd in range(1, 25):
                    futures.append(ex.submit(process_race, jcd, rno, today))
            concurrent.futures.wait(futures)

        # ★ ログを完全化
        log(f"🏁 判定完了: 対象={STATS['scanned']}R -> 見送={STATS['passed']}R, 購入={STATS['hits']}R "
            f"(待機={STATS['waiting']}R, 期限切={STATS['skipped']}R, エラー={STATS['errors']}R)")
        
        log("💤 180秒待機...")
        time.sleep(180)

    stop_event.set()

if __name__ == "__main__":
    main()
