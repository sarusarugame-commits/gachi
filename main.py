import os
import datetime
import time
import sqlite3
import concurrent.futures
import threading
import sys
import requests as std_requests
import json

from scraper import scrape_race_data, get_session, scrape_odds
from predict_boat import predict_race, attach_reason, load_model

DB_FILE = "race_data.db"
PLACE_NAMES = {i: n for i, n in enumerate(["","桐生","戸田","江戸川","平和島","多摩川","浜名湖","蒲郡","常滑","津","三国","びわこ","住之江","尼崎","鳴門","丸亀","児島","宮島","徳山","下関","若松","芦屋","福岡","唐津","大村"])}
JST = datetime.timezone(datetime.timedelta(hours=9), 'JST')

sys.stdout.reconfigure(encoding='utf-8')

DB_LOCK = threading.Lock()
# 統計用
STATS = {"scanned": 0, "hits": 0, "errors": 0, "skipped": 0}
STATS_LOCK = threading.Lock()

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
    log("ℹ️ レポート監視スレッド起動 (100円投資モード)")
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
                        # ★修正: 100円投資計算
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
                            msg = (
                                f"💀 **{p['place']}{p['race_no']}R** ハズレ\n"
                                f"予想: {combo} (結果: {result_str})\n"
                                f"📅 **本日トータル: {total_profit:+,}円**"
                            )
                            log(f"💀 ハズレ: {p['place']}{p['race_no']}R (結果:{result_str})")
                            send_discord(msg)
                conn.close()
        except Exception as e:
            pass
        
        for _ in range(10):
            if stop_event.is_set(): break
            time.sleep(60)

def process_race(jcd, rno, today):
    sess = get_session()
    place = PLACE_NAMES.get(jcd, "不明")
    
    # 1. データ取得
    try:
        raw, error = scrape_race_data(sess, jcd, rno, today)
    except Exception as e:
        with STATS_LOCK: STATS["errors"] += 1
        return

    if error or not raw:
        return

    # ★ 2. タイムフィルター (超重要)
    # 締切時刻をチェックし、「終わったレース」や「まだ先のレース」を弾く
    deadline_str = raw.get('deadline_time')
    
    if deadline_str:
        try:
            now = datetime.datetime.now(JST)
            h, m = map(int, deadline_str.split(':'))
            deadline_dt = now.replace(hour=h, minute=m, second=0, microsecond=0)
            
            # 日付またぎ対応（もし深夜レースなどで日付が変わる場合への保険）
            # 基本は当日比較でOK
            
            # 判定A: 既に終わっている（現在時刻 > 締切）
            if now > deadline_dt:
                # log(f"⏹️ {place}{rno}R: 終了済み (締切 {deadline_str})")
                with STATS_LOCK: STATS["skipped"] += 1
                return

            # 判定B: まだ先すぎる（締切 > 現在時刻 + 60分）
            # ※「朝に夜のレース通知はいらない」に対応
            if deadline_dt > (now + datetime.timedelta(minutes=60)):
                # log(f"⏳ {place}{rno}R: まだ先です (締切 {deadline_str})")
                with STATS_LOCK: STATS["skipped"] += 1
                return
                
        except:
            pass 
    else:
        # 締切時刻が取れない場合は、念のためチェックする（ログは出す）
        log(f"⚠️ {place}{rno}R: 締切時刻不明 -> 強制チェック")

    # 3. 予測実行
    try:
        preds = predict_race(raw)
    except Exception as e:
        with STATS_LOCK: STATS["errors"] += 1
        return

    with STATS_LOCK: STATS["scanned"] += 1

    if not preds:
        return

    # 4. DBチェック & 保存
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
    
    if not new_preds:
        return

    # 5. 解説生成 & 通知
    log(f"⚡ {place}{rno}R で {len(new_preds)}件の候補を検知！AI解説を生成中...")
    try:
        attach_reason(preds, raw)
    except Exception as e:
        log(f"⚠️ 解説生成エラー: {e}")

    with DB_LOCK:
        conn = sqlite3.connect(DB_FILE)
        for p in new_preds:
            combo = p['combo']
            race_id = f"{today}_{jcd}_{rno}_{combo}"
            
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
            with STATS_LOCK: STATS["hits"] += 1
            
        conn.close()

def main():
    log("🚀 最強AI Bot (本番運用モード v3.7) 起動 - 100円投資 & 直近レース厳選版")
    
    try:
        load_model()
        log("✅ AIモデル読み込み完了")
    except Exception as e:
        error_log(f"FATAL: モデル読み込みに失敗しました。\n詳細: {e}")
        error_log("強制終了します。")
        sys.exit(1)

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
        
        with STATS_LOCK:
            STATS["scanned"] = 0
            STATS["hits"] = 0
            STATS["errors"] = 0
            STATS["skipped"] = 0

        log(f"🔍 直近のレースをスキャン中 ({today})...")
        
        # ★修正: ループ順序を変更
        # 以前: 会場(1-24) -> レース(1-12) ※これだと24場の1Rが終わってから1場の2R...となる
        # 今回: レース(1-12) -> 会場(1-24) ※これなら全場の1Rを先にチェックできる
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
            futures = []
            for rno in range(1, 13):      # レース番号を外側のループに
                for jcd in range(1, 25):  # 会場を内側のループに
                    futures.append(ex.submit(process_race, jcd, rno, today))
            concurrent.futures.wait(futures)

        log(f"🏁 スキャン完了: 有効チェック={STATS['scanned']}, 範囲外スキップ={STATS['skipped']}, HIT={STATS['hits']}")
        log("💤 待機中(300秒)...")
        time.sleep(300)

    stop_event.set()

if __name__ == "__main__":
    main()
