import os
import datetime
import time
import sqlite3
import concurrent.futures
import threading
import sys
import requests as std_requests
import json

# scraper, predict_boat は同じフォルダに配置してください
from scraper import scrape_race_data, get_session, get_odds_map, get_odds_2t, scrape_result
from predict_boat import predict_race, attach_reason, load_models, filter_and_sort_bets, CONF_THRESH_3T, CONF_THRESH_2T, STRATEGY_3T, STRATEGY_2T, MIN_PROB_3T, check_groq_setup

DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "race_data.db")
PLACE_NAMES = {i: n for i, n in enumerate(["","桐生","戸田","江戸川","平和島","多摩川","浜名湖","蒲郡","常滑","津","三国","びわこ","住之江","尼崎","鳴門","丸亀","児島","宮島","徳山","下関","若松","芦屋","福岡","唐津","大村"])}
JST = datetime.timezone(datetime.timedelta(hours=9), 'JST')

sys.stdout.reconfigure(encoding='utf-8')

DB_LOCK = threading.Lock()
STATS = {"scanned": 0, "hits": 0, "errors": 0, "skipped": 0, "vetted": 0, "waiting": 0}
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
    except Exception as e:
        error_log(f"Discord通知エラー: {e}")

def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # テーブル作成（存在しない場合）
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            race_id TEXT PRIMARY KEY,
            date TEXT,
            place TEXT,
            race_no INTEGER,
            predict_combo TEXT,
            status TEXT,
            profit INTEGER,
            odds REAL,
            prob REAL,
            ev REAL,
            comment TEXT,
            ticket_type TEXT
        )
    """)
    
    # カラム不足の自動修復 (マイグレーション)
    cursor.execute("PRAGMA table_info(history)")
    columns = [row[1] for row in cursor.fetchall()]
    
    required_columns = {
        "odds": "REAL",
        "prob": "REAL",
        "ev": "REAL",
        "comment": "TEXT",
        "ticket_type": "TEXT",
        "result_odds": "REAL"
    }
    
    for col_name, col_type in required_columns.items():
        if col_name not in columns:
            try:
                print(f"🔄 DBマイグレーション: カラム '{col_name}' を追加します...")
                cursor.execute(f"ALTER TABLE history ADD COLUMN {col_name} {col_type}")
            except Exception as e:
                print(f"⚠️ マイグレーション警告: {e}")
    
    conn.commit()
    conn.close()

def report_worker(stop_event):
    log("ℹ️ レポート監視スレッド起動 (レース単位集約版)")
    while not stop_event.is_set():
        try:
            with DB_LOCK:
                conn = sqlite3.connect(DB_FILE)
                conn.row_factory = sqlite3.Row
                
                # 未完了のレースを会場・R番号でグルーピングして取得
                # race_id形式: YYYYMMDD_JCD_RNO_COMBO_TYPE
                pending_bets = conn.execute("SELECT * FROM history WHERE status='PENDING'").fetchall()
                if not pending_bets:
                    conn.close()
                    time.sleep(60)
                    continue

                # グルーピング: (date, jcd, rno) -> [bet_rows...]
                race_groups = {}
                for p in pending_bets:
                    try:
                        parts = p['race_id'].split('_')
                        jcd = int(parts[1])
                        key = (p['date'], jcd, p['race_no']) # date, jcd, rno
                        if key not in race_groups: race_groups[key] = []
                        race_groups[key].append(p)
                    except: continue
                
                if not race_groups:
                    conn.close()
                    time.sleep(60)
                    continue

                sess = get_session()
                
                for key, bets in race_groups.items():
                    date_str, jcd, rno = key
                    place_name = bets[0]['place']
                    
                    # 1. 結果取得 (レース単位で1回だけ)
                    res = scrape_result(sess, jcd, rno, date_str)
                    if not res: continue # まだ結果が出ていない
                    
                    # 2. まとめて判定 & DB更新
                    race_profit = 0
                    results_summary = []
                    hit_count = 0
                    
                    # 共通の結果文字列取得 (2t/3t両方確認)
                    res_str_2t = res.get('combo_2t', '未確定')
                    payout_2t = res.get('payout_2t', 0)
                    res_str_3t = res.get('combo_3t', '未確定')
                    payout_3t = res.get('payout_3t', 0)
                    
                    # 最終オッズ取得 (乖離計測用)
                    final_odds_2t = {}
                    final_odds_3t = {}
                    try:
                        final_odds_2t = get_odds_2t(sess, jcd, rno, date_str)
                        final_odds_3t = get_odds_map(sess, jcd, rno, date_str)
                    except Exception as e:
                        error_log(f"最終オッズ取得エラー: {e}")
                    
                    # 結果が両方とも未確定ならスキップ
                    if (res_str_2t == "未確定" or res_str_2t is None) and (res_str_3t == "未確定" or res_str_3t is None):
                        continue

                    # トランザクション更新
                    updated = False
                    
                    for bet in bets:
                        combo = bet['predict_combo']
                        t_type = bet['ticket_type']
                        race_id = bet['race_id']
                        
                        if t_type == '2t':
                            result_str = res_str_2t
                            payout = payout_2t
                        else:
                            result_str = res_str_3t
                            payout = payout_3t
                        
                        # 念のため結果が入っているか確認
                        if result_str == "未確定" or result_str is None:
                            continue # この券種の結果だけ出ていない等は稀だがスキップ

                        is_hit = (result_str == combo)
                        profit = payout - 100 if is_hit else -100
                        
                        # 最終オッズ取得
                        result_odds_val = 0.0
                        try:
                            if t_type == '2t':
                                result_odds_val = final_odds_2t.get(combo, 0.0)
                            else:
                                result_odds_val = final_odds_3t.get(combo, 0.0)
                        except: pass

                        conn.execute("UPDATE history SET status='FINISHED', profit=?, result_odds=? WHERE race_id=?", (profit, result_odds_val, race_id))
                        race_profit += profit
                        updated = True
                        
                        hit_mark = "🎯" if is_hit else "💀"
                        if is_hit: hit_count += 1
                        results_summary.append(f"{hit_mark} {combo} (結果:{result_str}) {'+' if profit>0 else ''}{profit}円")

                    if not updated: continue
                    conn.commit()
                    
                    # 3. 集計 & 通知作成
                    month_str = date_str[:6]
                    
                    total_profit_day = conn.execute("SELECT SUM(profit) FROM history WHERE date=? AND status='FINISHED'", (date_str,)).fetchone()[0] or 0
                    total_profit_month = conn.execute("SELECT SUM(profit) FROM history WHERE substr(date,1,6)=? AND status='FINISHED'", (month_str,)).fetchone()[0] or 0
                    
                    # 2連単成績
                    hits_2t = conn.execute("SELECT COUNT(*) FROM history WHERE date=? AND ticket_type='2t' AND status='FINISHED' AND profit > 0", (date_str,)).fetchone()[0]
                    total_2t = conn.execute("SELECT COUNT(*) FROM history WHERE date=? AND ticket_type='2t' AND status='FINISHED'", (date_str,)).fetchone()[0]
                    rate_2t = (hits_2t / total_2t * 100) if total_2t > 0 else 0.0

                    # 3連単成績
                    hits_3t = conn.execute("SELECT COUNT(*) FROM history WHERE date=? AND ticket_type='3t' AND status='FINISHED' AND profit > 0", (date_str,)).fetchone()[0]
                    total_3t = conn.execute("SELECT COUNT(*) FROM history WHERE date=? AND ticket_type='3t' AND status='FINISHED'", (date_str,)).fetchone()[0]
                    rate_3t = (hits_3t / total_3t * 100) if total_3t > 0 else 0.0
                    
                    # メッセージ構築
                    title_emoji = "🎉" if race_profit > 0 else "💀"
                    title_text = "的中！" if hit_count > 0 else "残念..."
                    
                    details = "\n".join(results_summary)
                    
                    msg = (
                        f"{title_emoji} **{place_name}{rno}R 結果** {title_text}\n"
                        f"{details}\n"
                        f"💰 レース収支: {'+' if race_profit>0 else ''}{race_profit:,}円\n"
                        f"-------------------\n"
                        f"📊 2連単: {hits_2t}/{total_2t} ({rate_2t:.1f}%)\n"
                        f"📊 3連単: {hits_3t}/{total_3t} ({rate_3t:.1f}%)\n"
                        f"📅 本日: {'+' if total_profit_day>0 else ''}{total_profit_day:,}円\n"
                        f"🗓️ 今月: {'+' if total_profit_month>0 else ''}{total_profit_month:,}円"
                    )
                    
                    log(f"📝 結果通知: {place_name}{rno}R (収支:{race_profit}円)")
                    send_discord(msg)
                    
                conn.close()
        except Exception as e:
            error_log(f"レポート監視エラー: {e}")
        time.sleep(60) # 頻度調整

def process_race(jcd, rno, today):
    try:
        with FINISHED_RACES_LOCK:
            if (jcd, rno) in FINISHED_RACES: return

        sess = get_session()
        place = PLACE_NAMES.get(jcd, "不明")
        
        try:
            raw, error = scrape_race_data(sess, jcd, rno, today)
        except Exception as e:
            with STATS_LOCK: STATS["errors"] += 1
            return

        if error != "OK" or not raw: return

        # 1. 時間管理 & 待機判定 (最優先)
        # まず対象会場かどうかチェック (Waitカウントのため)
        is_target = (jcd in STRATEGY_3T) or (jcd in STRATEGY_2T)
        if not is_target: return

        deadline_str = raw.get('deadline_time')
        if not deadline_str:
            log(f"⚠️ [スキップ] {place}{rno}R: 締切時間不明のため処理できません")
            with STATS_LOCK: STATS["errors"] += 1
            return

        try:
            now = datetime.datetime.now(JST)
            h, m = map(int, deadline_str.split(':'))
            deadline_dt = now.replace(hour=h, minute=m, second=0, microsecond=0)
            
            # 締切後 (1分経過)
            if now > (deadline_dt + datetime.timedelta(minutes=1)):
                with FINISHED_RACES_LOCK: FINISHED_RACES.add((jcd, rno))
                with STATS_LOCK: STATS["skipped"] += 1
                return

            # 締切15分前より前なら待機
            delta = deadline_dt - now
            if delta.total_seconds() > 900: 
                with STATS_LOCK: STATS["waiting"] += 1
                return
        except Exception as e:
            error_log(f"時間計算エラー {place}{rno}R: {e}")
            return

        # 2. 予測実行
        try:
            candidates, max_conf, max_removed_prob, _ = predict_race(raw)
        except Exception as e:
            error_log(f"予測エラー {place}{rno}R: {e}")
            with STATS_LOCK: STATS["errors"] += 1
            return

        # --- 見送り理由ログ: 自信度不足 ---
        if not candidates:
            # 3Tか2Tかによって閾値の表示を変える（簡易的に3T基準で表示、または高い方）
            thresh_display = max(CONF_THRESH_3T, CONF_THRESH_2T)
            min_prob_display = MIN_PROB_3T # 厳密には2T等あるが代表値として
            
            if max_conf > 0:
                if max_conf < thresh_display:
                    log(f"👀 [見送り] {place}{rno}R: 自信度不足 (AIスコア:{max_conf:.2f} < 基準:{thresh_display})")
                else:
                    # 自信度は足りているが、個別の買い目確率が基準(MIN_PROB)に届かなかった場合
                    log(f"👀 [見送り] {place}{rno}R: 組み合わせ確率不足 (AIスコア:{max_conf:.2f}OK 最大コンボ:{max_removed_prob*100:.1f}% < 基準:{min_prob_display*100:.0f}%)")
            
            with STATS_LOCK: STATS["vetted"] += 1
            return

        # 3. オッズ取得
        odds_2t, odds_3t = {}, {}
        has_2t = any(c['type'] == '2t' for c in candidates)
        has_3t = any(c['type'] == '3t' for c in candidates)
        
        try:
            if has_2t: odds_2t = get_odds_2t(sess, jcd, rno, today)
            if has_3t: odds_3t = get_odds_map(sess, jcd, rno, today)
        except Exception as e:
            error_log(f"オッズ取得例外 {place}{rno}R: {e}")

        # 4. EVフィルタリング
        try:
            final_bets, max_ev, current_thresh = filter_and_sort_bets(candidates, odds_2t, odds_3t, jcd)
        except: return

        # --- 見送り理由ログ: 期待値(EV)不足 ---
        if not final_bets:
            # 候補はあったが、オッズと掛け合わせたら期待値が足りなかった場合
            if max_ev > 0:
                log(f"📉 [見送り] {place}{rno}R: 期待値不足 (最大EV:{max_ev:.2f} < 基準:{current_thresh})")
            else:
                log(f"📉 [見送り] {place}{rno}R: オッズ取得失敗または有効オッズなし")
            
            with STATS_LOCK: STATS["vetted"] += 1
            return

        # 5. 解説生成
        try:
            attach_reason(final_bets, raw, {})
        except Exception as e:
            error_log(f"解説生成エラー: {e}")

        # 6. DB保存 & 通知
        with STATS_LOCK: STATS["scanned"] += 1
        with DB_LOCK:
            conn = sqlite3.connect(DB_FILE)
            for p in final_bets:
                combo = p['combo']
                t_type = p['type']
                race_id = f"{today}_{jcd}_{rno}_{combo}_{t_type}"
                
                if conn.execute("SELECT 1 FROM history WHERE race_id=?", (race_id,)).fetchone(): continue

                prob = float(p.get('prob', 0))
                reason = p.get('reason', '解説取得失敗')
                odds_val = p.get('odds', 0.0)
                ev_val = p.get('ev', 0.0)
                
                log(f"🔥 [HIT] {place}{rno}R ({t_type.upper()}) -> {combo} ({odds_val}倍 EV:{ev_val:.2f})")
                
                odds_url = f"https://www.boatrace.jp/owpc/pc/race/odds{'2tf' if t_type=='2t' else '3t'}?rno={rno}&jcd={jcd:02d}&hd={today}"
                deadline_str = raw.get('deadline_time', '不明')

                msg = (
                    f"🔥 **{place}{rno}R** {t_type.upper()}激アツ (締切: {deadline_str})\n"
                    f"🎯 買い目: **{combo}**\n"
                    f"📊 確率: **{prob}%** / オッズ: **{odds_val}倍**\n"
                    f"💎 期待値: **{ev_val:.2f}**\n"
                    f"📝 AI寸評: {reason}\n"
                    f"🔗 [オッズ確認]({odds_url})"
                )
                
                conn.execute(
                    "INSERT INTO history VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (race_id, today, place, rno, combo, 'PENDING', 0, odds_val, prob, ev_val, reason, t_type, 0.0)
                )
                conn.commit()
                
                # 書き込み確認ログ
                try:
                    current_count = conn.execute("SELECT COUNT(*) FROM history").fetchone()[0]
                    log(f"💾 DB保存完了 [Total: {current_count}件] ID:{race_id}")
                except:
                    pass

                send_discord(msg)
                with STATS_LOCK: STATS["hits"] += 1
            conn.close()
    except Exception as e:
        import traceback
        error_log(f"CRITICAL ERROR in process_race ({place}{rno}R): {e}")
        error_log(traceback.format_exc())

def main():
    log(f"🚀 ハイブリッドAI Bot (ROI130% & 黄金律) 起動")
    
    try:
        check_groq_setup()
        load_models()
        log("✅ AIモデル(2T/3T) 読み込み完了")
    except Exception as e:
        error_log(f"FATAL: モデル読み込みエラー: {e}")
        sys.exit(1)

    init_db()
    
    # 🐞 DBパスとレコード数確認用
    try:
        abs_db_path = os.path.abspath(DB_FILE)
        log(f"📁 DBファイル絶対パス: {abs_db_path}")
        if os.path.exists(abs_db_path):
            conn = sqlite3.connect(DB_FILE)
            cnt = conn.execute("SELECT COUNT(*) FROM history").fetchone()[0]
            conn.close()
            log(f"📊 現在のDBレコード数: {cnt}件")
            log(f"📉 DBファイルサイズ: {os.path.getsize(abs_db_path)} bytes")
        else:
            log("⚠️ DBファイルがまだ存在しません")
    except Exception as e:
        error_log(f"DBチェックエラー: {e}")
    
    # Discord設定確認
    if os.environ.get("DISCORD_WEBHOOK_URL"):
        log("ℹ️ Discord通知: ON")
    else:
        log("⚠️ Discord通知: OFF (環境変数が設定されていません)")

    stop_event = threading.Event()
    t = threading.Thread(target=report_worker, args=(stop_event,), daemon=True)
    t.start()
    
    start_time = time.time()
    MAX_RUNTIME = 20700 # 5時間45分 (GitHub Actions 6時間制限回避のため)
    
    while True:
        if time.time() - start_time > MAX_RUNTIME:
            log("🔄 稼働時間上限のため終了")
            break
        
        now = datetime.datetime.now(JST)
        
        # 夜間停止 (22:00 〜 08:00 は停止)
        if now.hour >= 22 or now.hour < 8:
            log(f"🌙 夜間のため稼働を終了します ({now.strftime('%H:%M')})")
            break
            
        today = now.strftime('%Y%m%d')
        
        # 統計リセット
        with STATS_LOCK:
            STATS["scanned"] = 0; STATS["hits"] = 0; STATS["errors"] = 0
            STATS["skipped"] = 0; STATS["vetted"] = 0; STATS["waiting"] = 0

        log(f"🔍 スキャン開始 ({today})...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
            for rno in range(1, 13):
                for jcd in range(1, 25):
                    ex.submit(process_race, jcd, rno, today)

        log(f"🏁 サイクル完了: 購入={STATS['hits']}, 見送り={STATS['vetted']}, 待機={STATS['waiting']}, 締切={STATS['skipped']}")
        time.sleep(60)

    stop_event.set()

if __name__ == "__main__":
    main()
