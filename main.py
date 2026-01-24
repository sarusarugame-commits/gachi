import os
import datetime
import time
import pandas as pd
import numpy as np
import lightgbm as lgb
import requests
import sqlite3
import concurrent.futures
import zipfile
import traceback
import threading
import re
from collections import defaultdict

# scraper.py から必要な機能をすべてインポート
from scraper import scrape_race_data, scrape_odds, scrape_result

# ==========================================
# ⚙️ 設定エリア
# ==========================================
DB_FILE = "race_data.db"
BET_AMOUNT = 1000  # 1点あたりの購入額

# 🤖 予測フィルター設定
# モデルが過信気味なため、確率だけでなく「期待値(EV)」も条件に追加
THRESHOLD_NIRENTAN = 0.15  # 2連単の確率しきい値
THRESHOLD_TANSHO   = 0.40  # 単勝の確率しきい値
MIN_EV             = 1.0   # 期待値しきい値（1.0未満は買わない）

REPORT_HOURS = list(range(8, 24))

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
# 日本語対応・指示従順性が高いモデルを指定
GROQ_MODEL_NAME = "llama3-70b-8192" 

MODEL_FILE = 'boat_model_nirentan.txt'
ZIP_MODEL = 'model.zip'
COMBOS = [f"{f}-{s}" for f in range(1, 7) for s in range(1, 7) if f != s]
PLACE_NAMES = {
    1: "桐生", 2: "戸田", 3: "江戸川", 4: "平和島", 5: "多摩川", 6: "浜名湖",
    7: "蒲郡", 8: "常滑", 9: "津", 10: "三国", 11: "びわこ", 12: "住之江",
    13: "尼崎", 14: "鳴門", 15: "丸亀", 16: "児島", 17: "宮島", 18: "徳山",
    19: "下関", 20: "若松", 21: "芦屋", 22: "福岡", 23: "唐津", 24: "大村"
}

t_delta = datetime.timedelta(hours=9)
JST = datetime.timezone(t_delta, 'JST')

IGNORE_RACES = set()

# ==========================================
# 🛠️ ユーティリティ & API
# ==========================================
def extract_odds_value(odds_text, target_boat=None):
    try:
        if re.match(r"^\d+\.\d+$", str(odds_text)): return float(odds_text)
        match = re.search(r"(\d+\.\d+)", str(odds_text))
        if match: return float(match.group(1))
    except: pass
    return 0.0

def call_groq_api(prompt):
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key: return "APIキー未設定"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    messages = [
        {
            "role": "system",
            "content": (
                "あなたは日本のボートレース予想記者です。"
                "提供されたデータを元に、推奨理由を一言（日本語40文字以内）で述べてください。"
                "英語の解説、挨拶、分析の過程は一切出力しないでください。"
                "出力は推奨コメントのみにしてください。"
            )
        },
        {
            "role": "user",
            "content": f"データ: {prompt}\nこのデータの推奨理由を日本語40文字以内で書いて。"
        }
    ]
    
    data = {
        "model": GROQ_MODEL_NAME,
        "messages": messages,
        "temperature": 0.3, 
        "max_tokens": 60
    }
    
    try:
        res = requests.post(GROQ_API_URL, headers=headers, json=data, timeout=30)
        if res.status_code == 200:
            content = res.json()['choices'][0]['message']['content']
            return content.replace("\n", "").replace('"', '').replace("`", "").strip()
        else:
            return "応答エラー"
    except: return "応答なし"

def send_discord(content):
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not url: return
    try: requests.post(url, json={"content": content}, timeout=10)
    except: pass

def get_db_connection():
    conn = sqlite3.connect(DB_FILE, timeout=60, isolation_level=None)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("PRAGMA journal_mode=WAL;")
    
    c.execute('''CREATE TABLE IF NOT EXISTS history (
        race_id TEXT PRIMARY KEY, date TEXT, time TEXT, place TEXT, race_no INTEGER,
        predict_combo TEXT, predict_prob REAL, gemini_comment TEXT,
        result_combo TEXT, is_win INTEGER, payout INTEGER, profit INTEGER, status TEXT
    )''')
    
    required_cols = {'best_boat': 'TEXT', 'odds_tansho': 'TEXT', 'odds_nirentan': 'TEXT', 'result_tansho': 'TEXT'}
    try:
        c.execute("PRAGMA table_info(history)")
        existing_cols = {row['name'] for row in c.fetchall()}
        for col, dtype in required_cols.items():
            if col not in existing_cols:
                c.execute(f"ALTER TABLE history ADD COLUMN {col} {dtype}")
    except: pass
    conn.close()

# ==========================================
# 📊 報告専用スレッド
# ==========================================
def report_worker():
    print("📋 [Report] 報告スレッド起動")
    
    while True:
        try:
            now = datetime.datetime.now(JST)
            today = now.strftime('%Y%m%d')
            conn = get_db_connection()
            c = conn.cursor()
            
            # PENDING（購入済み・未確定）のレースを確認
            c.execute("SELECT * FROM history WHERE status='PENDING'")
            pending_races = c.fetchall()
            
            # レースごとにグルーピング (YYYYMMDD_JJ_RR をキーにする)
            races_by_id = defaultdict(list)
            for race in pending_races:
                base_id = "_".join(race['race_id'].split('_')[:3])
                races_by_id[base_id].append(race)
            
            sess = requests.Session()
            updates = 0
            
            for base_id, race_list in races_by_id.items():
                try:
                    # 代表データの取得
                    first_race = race_list[0]
                    date_str = first_race['date']
                    
                    # IDから会場コードとレース番号を復元
                    parts = base_id.split('_')
                    jcd_int = int(parts[1])
                    rno_int = int(parts[2])
                    
                    formatted_date = f"{date_str[:4]}/{date_str[4:6]}/{date_str[6:]}"
                    place_name = PLACE_NAMES.get(jcd_int, "会場")

                    # 結果スクレイピング
                    res = scrape_result(sess, jcd_int, rno_int, date_str)
                    
                    if not res: continue

                    # レース結果情報の準備
                    nirentan_res = res['nirentan_combo']
                    nirentan_pay = res['nirentan_payout']
                    tansho_res = res['tansho_boat']
                    tansho_pay = res['tansho_payout']
                    
                    if not (nirentan_res or tansho_res): continue

                    race_profit = 0
                    results_text = []
                    is_any_win = False
                    
                    # グループ内の各チケットを処理
                    for race in race_list:
                        pred_combo = race['predict_combo'] 
                        is_win = 0
                        actual_result = ""
                        payout_per_100 = 0
                        type_lbl = ""
                        
                        if "-" in str(pred_combo): # 2連単
                            type_lbl = "2単"
                            actual_result = nirentan_res
                            payout_per_100 = nirentan_pay
                        else: # 単勝
                            type_lbl = "単勝"
                            actual_result = tansho_res
                            payout_per_100 = tansho_pay

                        # 勝敗判定と収支計算（ここを修正！）
                        profit = -BET_AMOUNT # 外れの場合のデフォルト
                        
                        if str(pred_combo) == str(actual_result):
                            is_win = 1
                            # 払戻金計算: (オッズ × 購入額/100) - 購入額
                            # 例: 230円 * (1000/100) - 1000 = 2300 - 1000 = +1300
                            bet_ratio = BET_AMOUNT / 100
                            return_amount = int(payout_per_100 * bet_ratio)
                            profit = return_amount - BET_AMOUNT
                        
                        # DB更新
                        c.execute("""
                            UPDATE history 
                            SET result_combo=?, is_win=?, payout=?, profit=?, status='FINISHED', result_tansho=?
                            WHERE race_id=?
                        """, (actual_result, is_win, payout_per_100, profit, tansho_res, race['race_id']))
                        updates += 1
                        
                        race_profit += profit
                        if is_win: is_any_win = True
                        
                        # 結果行の作成
                        icon = "🎯" if is_win else "💀"
                        results_text.append(f"{icon} **{type_lbl}**: {pred_combo} (収支: {profit:+d}円)")

                    # 累計計算
                    c.execute("SELECT sum(profit) FROM history WHERE date=? AND status='FINISHED'", (today,))
                    daily_profit = c.fetchone()[0] or 0
                    
                    # 通知
                    header_icon = "🎉" if race_profit > 0 else "📢"
                    msg = (f"{header_icon} **{formatted_date} {place_name}{rno_int}R 結果**\n"
                           f"🏁 結果: {nirentan_res} (単: {tansho_res})\n"
                           + "\n".join(results_text) + "\n"
                           f"💰 レース収支: {race_profit:+d}円\n"
                           f"📉 本日累計: {daily_profit:+d}円")
                    send_discord(msg)
                    print(f"📊 [Report] 判明: {place_name}{rno_int}R 収支:{race_profit}")
                    
                    time.sleep(1)
                except Exception as e:
                    print(f"Report Group Error: {e}")
                    continue
            
            if updates > 0: print(f"✅ [Report] {updates}件更新")
            conn.close()
        except Exception as e:
            print(f"🔥 [Report] Error: {e}")
            traceback.print_exc()
        
        time.sleep(300)

# ==========================================
# 🚤 予想ロジック
# ==========================================
def engineer_features(df):
    for i in range(1, 7): df[f'power_idx_{i}'] = df[f'wr{i}'] * (1.0 / (df[f'st{i}'] + 0.01))
    for i in range(1, 6):
        df[f'st_gap_{i}_{i+1}'] = df[f'st{i+1}'] - df[f'st{i}']
        df[f'wr_gap_{i}_{i+1}'] = df[f'wr{i}'] - df[f'wr{i+1}']
    avg_wr = df[[f'wr{i}' for i in range(1, 7)]].mean(axis=1)
    df['wr_1_vs_avg'] = df['wr1'] / (avg_wr + 0.001)
    df['jcd'] = df['jcd'].astype('category')
    return df

def calculate_tansho(probs):
    win = {i: 0.0 for i in range(1, 7)}
    for idx, c in enumerate(COMBOS): win[int(c.split('-')[0])] += probs[idx]
    return win

def is_target_race(deadline_str, now_dt):
    try:
        if not deadline_str or deadline_str == "23:59": return True
        hm = deadline_str.split(":")
        d_dt = now_dt.replace(hour=int(hm[0]), minute=int(hm[1]), second=0)
        if d_dt < now_dt - datetime.timedelta(hours=1): d_dt += datetime.timedelta(days=1)
        if now_dt > d_dt: return False
        return (d_dt - now_dt) <= datetime.timedelta(minutes=60)
    except: return True

def get_odds_with_retry(sess, jcd, rno, today, best_b, combo):
    for _ in range(3):
        odds_data = scrape_odds(sess, jcd, rno, today, target_boat=str(best_b), target_combo=combo)
        if odds_data['tansho'] != "---": return odds_data
        time.sleep(2)
    return {"tansho": "1.0", "nirentan": "1.0"}

def process_prediction(jcd, today, notified_ids, bst):
    global IGNORE_RACES
    pred_list = []
    sess = requests.Session()
    now = datetime.datetime.now(JST)
    
    conn_temp = get_db_connection()
    c_temp = conn_temp.cursor()
    c_temp.execute("SELECT sum(profit) FROM history WHERE date=? AND status='FINISHED'", (today,))
    current_daily_profit = c_temp.fetchone()[0] or 0
    conn_temp.close()
    
    for rno in range(1, 13):
        base_rid = f"{today}_{str(jcd).zfill(2)}_{rno}"
        if base_rid in IGNORE_RACES: continue

        rid_tansho = f"{base_rid}_T"
        rid_nirentan = f"{base_rid}_N"
        
        # 既に両方通知済みならスキップ
        if rid_tansho in notified_ids and rid_nirentan in notified_ids: continue
        
        try:
            raw = scrape_race_data(sess, jcd, rno, today)
            if not raw: IGNORE_RACES.add(base_rid); continue
            if not is_target_race(raw.get('deadline_time'), now): IGNORE_RACES.add(base_rid); continue
            
            df = engineer_features(pd.DataFrame([raw]))
            cols = ['jcd', 'rno', 'wind', 'wr_1_vs_avg']
            for i in range(1, 7): cols.extend([f'wr{i}', f'st{i}', f'ex{i}', f'power_idx_{i}'])
            for i in range(1, 6): cols.extend([f'st_gap_{i}_{i+1}', f'wr_gap_{i}_{i+1}'])
            
            probs = bst.predict(df[cols])[0]
            win_p = calculate_tansho(probs)
            best_b = max(win_p, key=win_p.get)
            best_idx = np.argmax(probs)
            combo, prob = COMBOS[best_idx], probs[best_idx]

            odds_data = get_odds_with_retry(sess, jcd, rno, today, best_b, combo)
            real_odds_t = extract_odds_value(odds_data['tansho'])
            real_odds_n = extract_odds_value(odds_data['nirentan'])
            if real_odds_t == 0: real_odds_t = 1.0
            if real_odds_n == 0: real_odds_n = 1.0

            # --- 予測とフィルタリング ---
            
            # 1. 単勝 (確率 > 40% かつ EV > 1.0)
            if rid_tansho not in notified_ids and win_p[best_b] >= THRESHOLD_TANSHO:
                ev_t = real_odds_t * win_p[best_b]
                
                # EVチェックを追加
                if ev_t >= MIN_EV:
                    comment = call_groq_api(f"単勝{best_b}。期待値{ev_t:.2f}。")
                    pred_list.append({
                        'id': rid_tansho, 'jcd': jcd, 'rno': rno, 'date': today,
                        'combo': str(best_b), 'prob': win_p[best_b], 'best_boat': best_b,
                        'comment': comment, 'deadline': raw.get('deadline_time'),
                        'odds': odds_data, 'ev': ev_t, 'type': '単勝'
                    })

            # 2. 2連単 (確率 > 15% かつ EV > 1.0)
            if rid_nirentan not in notified_ids and prob >= THRESHOLD_NIRENTAN:
                ev_n = real_odds_n * prob
                
                # EVチェックを追加
                if ev_n >= MIN_EV:
                    comment = call_groq_api(f"2連単{combo}。期待値{ev_n:.2f}。")
                    pred_list.append({
                        'id': rid_nirentan, 'jcd': jcd, 'rno': rno, 'date': today,
                        'combo': combo, 'prob': prob, 'best_boat': best_b,
                        'comment': comment, 'deadline': raw.get('deadline_time'),
                        'odds': odds_data, 'ev': ev_n, 'type': '2単'
                    })
            
        except: continue
    
    return pred_list, current_daily_profit, f"{today[:4]}/{today[4:6]}/{today[6:]}"

def main():
    print(f"🚀 [Main] 収支計算修正版Bot起動 (Model: {GROQ_MODEL_NAME})")
    init_db()
    
    if not os.path.exists(MODEL_FILE):
        if not os.path.exists(ZIP_MODEL):
            if os.path.exists('model_part_1'):
                with open(ZIP_MODEL, 'wb') as f_out:
                    for i in range(1, 20):
                        p = f'model_part_{i}' if os.path.exists(f'model_part_{i}') else f'model_part_{i:02d}'
                        if os.path.exists(p): 
                            with open(p, 'rb') as f_in: f_out.write(f_in.read())
                        else: break
        if os.path.exists(ZIP_MODEL):
            with zipfile.ZipFile(ZIP_MODEL, 'r') as f: f.extractall()
    
    try: bst = lgb.Booster(model_file=MODEL_FILE)
    except: return

    t = threading.Thread(target=report_worker, daemon=True)
    t.start()
    start_ts = time.time()

    while True:
        now = datetime.datetime.now(JST)
        today = now.strftime('%Y%m%d')
        if now.hour >= 23 and now.minute >= 10: break
        if time.time() - start_ts > 21000: break

        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT race_id FROM history")
        notified_ids = set(row[0] for row in c.fetchall())
        conn.close()

        print(f"⚡️ [Main] スキャン: {now.strftime('%H:%M:%S')}")
        
        new_preds = []
        current_daily_profit = 0
        formatted_date = today
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            futures = [executor.submit(process_prediction, jcd, today, notified_ids, bst) for jcd in range(1, 25)]
            for f in concurrent.futures.as_completed(futures):
                try: 
                    res, profit, date_fmt = f.result()
                    new_preds.extend(res)
                    current_daily_profit = profit
                    formatted_date = date_fmt
                except: pass
        
        if new_preds:
            conn = get_db_connection()
            c = conn.cursor()
            
            # レースごとのグループ通知（購入時）
            preds_by_race = defaultdict(list)
            for pred in new_preds:
                preds_by_race[(pred['jcd'], pred['rno'])].append(pred)
            
            for (jcd, rno), preds in preds_by_race.items():
                try:
                    now_str = datetime.datetime.now(JST).strftime('%H:%M:%S')
                    place_name = PLACE_NAMES.get(jcd, "不明")
                    first_pred = preds[0]
                    t_disp = f"(締切 {first_pred['deadline']})" if first_pred['deadline'] else ""
                    odds_url = f"https://www.boatrace.jp/owpc/pc/race/oddstf?rno={rno}&jcd={jcd:02d}&hd={today}"
                    
                    details_text = []
                    
                    # DB保存とメッセージ行作成
                    for pred in preds:
                        c.execute("""
                            INSERT OR IGNORE INTO history 
                            (race_id, date, time, place, race_no, predict_combo, predict_prob, gemini_comment, 
                             result_combo, is_win, payout, profit, status, best_boat, odds_tansho, odds_nirentan, result_tansho)
                            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """, (
                            pred['id'], pred['date'], now_str, place_name, pred['rno'], pred['combo'], float(pred['prob']), pred['comment'], 
                            "", 0, 0, 0, "PENDING", str(pred['best_boat']), pred['odds']['tansho'], pred['odds']['nirentan'], ""
                        ))
                        
                        type_str = pred['type']
                        odds_val = pred['odds']['tansho'] if type_str == "単勝" else pred['odds']['nirentan']
                        ev_val = pred.get('ev', 0.0)
                        
                        details_text.append(
                            f"🎫 **{type_str}**: {pred['combo']} (率:{pred['prob']:.0%} / オッズ:{odds_val} / EV:{ev_val:.2f})"
                        )

                    # コメントは代表して1つ
                    comment_disp = first_pred['comment']

                    msg = (f"🔥 **{formatted_date} {place_name}{rno}R** {t_disp}\n"
                           f"🛶 本命: {first_pred['best_boat']}号艇\n"
                           + "\n".join(details_text) + "\n"
                           f"━━━━━━━━━━━━━━\n"
                           f"🤖 {comment_disp}\n"
                           f"━━━━━━━━━━━━━━\n"
                           f"📉 本日累計: {'+' if current_daily_profit>0 else ''}{current_daily_profit}円\n"
                           f"📊 [オッズ]({odds_url})")
                    send_discord(msg)
                    print(f"✅ [Main] 通知: {place_name}{rno}R")
                    
                except Exception as e:
                    print(f"Insert Error: {e}")
            conn.close()

        elapsed = time.time() - start_ts
        time.sleep(max(0, 180 - elapsed % 180))

if __name__ == "__main__":
    main()
