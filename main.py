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

# scraper.py から必要な機能をすべてインポート
from scraper import scrape_race_data, scrape_odds, scrape_result

# ==========================================
# ⚙️ 設定エリア
# ==========================================
DB_FILE = "race_data.db"
BET_AMOUNT = 1000

# 閾値
THRESHOLD_NIRENTAN = 0.15
THRESHOLD_TANSHO   = 0.40

REPORT_HOURS = list(range(8, 24))

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"

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
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {"model": GROQ_MODEL_NAME, "messages": [{"role": "user", "content": prompt}], "temperature": 0.3}
    try:
        res = requests.post(GROQ_API_URL, headers=headers, json=data, timeout=30)
        if res.status_code == 200: return res.json()['choices'][0]['message']['content']
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
    
    # マイグレーション
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
            
            c.execute("SELECT * FROM history WHERE status='PENDING'")
            pending_races = c.fetchall()
            
            sess = requests.Session()
            updates = 0
            
            for race in pending_races:
                try:
                    parts = race['race_id'].split('_')
                    # ID形式: YYYYMMDD_JJ_RR_TYPE or YYYYMMDD_JJ_RR
                    date_str = parts[0]
                    jcd = int(parts[1])
                    rno = int(parts[2])
                    
                    formatted_date = f"{date_str[:4]}/{date_str[4:6]}/{date_str[6:]}"
                    
                    res = scrape_result(sess, jcd, rno, date_str)
                    
                    if res:
                        pred_combo = race['predict_combo'] 
                        is_win = 0
                        profit = -BET_AMOUNT
                        actual_result = ""
                        payout = 0
                        
                        nirentan_res = res['nirentan_combo']
                        nirentan_pay = res['nirentan_payout']
                        tansho_res = res['tansho_boat']
                        tansho_pay = res['tansho_payout']
                        
                        # 判定
                        if "-" in str(pred_combo): # 2連単
                            actual_result = nirentan_res
                            payout = nirentan_pay
                            if str(pred_combo) == str(actual_result):
                                is_win = 1
                                profit = payout - BET_AMOUNT
                        else: # 単勝
                            actual_result = tansho_res
                            payout = tansho_pay
                            if str(pred_combo) == str(actual_result):
                                is_win = 1
                                profit = payout - BET_AMOUNT

                        if not actual_result: continue

                        c.execute("""
                            UPDATE history 
                            SET result_combo=?, is_win=?, payout=?, profit=?, status='FINISHED', result_tansho=?
                            WHERE race_id=?
                        """, (actual_result, is_win, payout, profit, tansho_res, race['race_id']))
                        updates += 1
                        
                        # 累計計算
                        c.execute("SELECT sum(profit) FROM history WHERE date=? AND status='FINISHED'", (today,))
                        daily_profit = c.fetchone()[0] or 0
                        
                        place = PLACE_NAMES.get(jcd, "会場")
                        type_lbl = "2単" if "-" in str(pred_combo) else "単勝"
                        
                        msg = (f"{'🎊 的中' if is_win else '💀 外れ'} {formatted_date} {place}{rno}R ({type_lbl})\n"
                               f"予測:{pred_combo} → 結果:{actual_result} (単:{tansho_res})\n"
                               f"収支:{'+' if profit>0 else ''}{profit}円\n"
                               f"📉 本日累計: {'+' if daily_profit>0 else ''}{daily_profit}円")
                        send_discord(msg)
                        print(f"📊 [Report] 判明: {place}{rno}R")
                    
                    time.sleep(1)
                except: continue
            
            if updates > 0: print(f"✅ [Report] {updates}件更新")

            # 定期報告
            current_key = f"{today}_{now.hour}"
            if now.hour in REPORT_HOURS and last_report_key != current_key:
                # 報告処理... (省略せず実装)
                c.execute("SELECT count(*), sum(is_win), sum(profit) FROM history WHERE date=? AND status='FINISHED'", (today,))
                cnt, wins, profit = c.fetchone()
                c.execute("SELECT count(*) FROM history WHERE status='PENDING'")
                pending_cnt = c.fetchone()[0]
                
                msg = (f"**🛠️ {now.hour}時の定期報告 ({today})**\n"
                       f"✅ 判明: {cnt or 0}R (的中: {wins or 0})\n"
                       f"⏳ 待ち: {pending_cnt or 0}R\n"
                       f"💵 本日収支: {'+' if (profit or 0)>0 else ''}{profit or 0}円")
                send_discord(msg)
                last_report_key = current_key

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
    
    # 日付整形
    formatted_date = f"{today[:4]}/{today[4:6]}/{today[6:]}"
    
    # 累計収支取得
    conn_temp = get_db_connection()
    c_temp = conn_temp.cursor()
    c_temp.execute("SELECT sum(profit) FROM history WHERE date=? AND status='FINISHED'", (today,))
    current_daily_profit = c_temp.fetchone()[0] or 0
    conn_temp.close()
    
    for rno in range(1, 13):
        # レースごとのベースID
        base_rid = f"{today}_{str(jcd).zfill(2)}_{rno}"
        
        # 既にチェック済みならスキップ (末尾_N, _Tは別判定)
        if base_rid in IGNORE_RACES: continue

        # 既に通知済みのIDかをチェック (単勝と2連単それぞれ)
        rid_tansho = f"{base_rid}_T"
        rid_nirentan = f"{base_rid}_N"
        
        # 両方とも通知済みならスキップ
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

            # オッズ取得 (1回で済ませる)
            odds_data = get_odds_with_retry(sess, jcd, rno, today, best_b, combo)
            real_odds_t = extract_odds_value(odds_data['tansho'])
            real_odds_n = extract_odds_value(odds_data['nirentan'])
            if real_odds_t == 0: real_odds_t = 1.0
            if real_odds_n == 0: real_odds_n = 1.0

            # --- 単勝の判定 ---
            if rid_tansho not in notified_ids and win_p[best_b] >= THRESHOLD_TANSHO:
                ev_t = real_odds_t * win_p[best_b]
                prompt = f"単勝{best_b}。EV:{ev_t:.2f}。基準1.0以上で買い。40文字以内。"
                comment = call_groq_api(prompt)
                
                pred_list.append({
                    'id': rid_tansho, 'jcd': jcd, 'rno': rno, 'date': today,
                    'combo': str(best_b), 'prob': win_p[best_b], 'best_boat': best_b,
                    'win_prob': win_p[best_b], 'comment': comment,
                    'deadline': raw.get('deadline_time'), 'odds': odds_data, 'ev': ev_t,
                    'type': '単勝'
                })

            # --- 2連単の判定 ---
            if rid_nirentan not in notified_ids and prob >= THRESHOLD_NIRENTAN:
                ev_n = real_odds_n * prob
                prompt = f"2単{combo}。EV:{ev_n:.2f}。基準1.0以上で買い。40文字以内。"
                comment = call_groq_api(prompt)

                pred_list.append({
                    'id': rid_nirentan, 'jcd': jcd, 'rno': rno, 'date': today,
                    'combo': combo, 'prob': prob, 'best_boat': best_b,
                    'win_prob': win_p[best_b], 'comment': comment,
                    'deadline': raw.get('deadline_time'), 'odds': odds_data, 'ev': ev_n,
                    'type': '2単'
                })
            
            # 条件に合わなければ、次回からスキップするためにリストに入れる
            if not pred_list:
                # 今回どちらも引っかからなかった場合のみ（保留）
                # 実際にはオッズ変動で変わるかもしれないので、本来はスキップしない方がいいが、負荷軽減のため
                # IGNORE_RACES.add(base_rid) 
                pass

        except: continue
    
    return pred_list, current_daily_profit, formatted_date

def main():
    print(f"🚀 [Main] 完全統合Bot起動 (Model: {GROQ_MODEL_NAME})")
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
            for pred in new_preds:
                try:
                    now_str = datetime.datetime.now(JST).strftime('%H:%M:%S')
                    place = PLACE_NAMES.get(pred['jcd'], "不明")
                    
                    c.execute("""
                        INSERT OR IGNORE INTO history 
                        (race_id, date, time, place, race_no, predict_combo, predict_prob, gemini_comment, 
                         result_combo, is_win, payout, profit, status, best_boat, odds_tansho, odds_nirentan, result_tansho)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """, (
                        pred['id'], pred['date'], now_str, place, pred['rno'], pred['combo'], float(pred['prob']), pred['comment'], 
                        "", 0, 0, 0, "PENDING", str(pred['best_boat']), pred['odds']['tansho'], pred['odds']['nirentan'], ""
                    ))
                    
                    t_disp = f"(締切 {pred['deadline']})" if pred['deadline'] else ""
                    odds_url = f"https://www.boatrace.jp/owpc/pc/race/oddstf?rno={pred['rno']}&jcd={pred['jcd']:02d}&hd={pred['date']}"
                    
                    type_str = pred['type'] # "単勝" or "2単"
                    odds_val = pred['odds']['tansho'] if type_str == "単勝" else pred['odds']['nirentan']
                    ev_val = pred.get('ev', 0.0)

                    msg = (f"🔥 **{formatted_date} {place}{pred['rno']}R** {t_disp}\n"
                           f"🛶 本命: {pred['best_boat']}号艇\n"
                           f"🎯 推奨: {pred['combo']} ({type_str}/率:{pred['prob']:.0%})\n"
                           f"💰 オッズ: {odds_val}\n"
                           f"📈 期待値: {ev_val:.2f}\n"
                           f"━━━━━━━━━━━━━━\n"
                           f"🤖 **{pred['comment']}**\n"
                           f"━━━━━━━━━━━━━━\n"
                           f"📉 本日累計: {'+' if current_daily_profit>0 else ''}{current_daily_profit}円\n"
                           f"📊 [オッズ]({odds_url})")
                    send_discord(msg)
                    print(f"✅ [Main] 通知: {place}{pred['rno']}R")
                except Exception as e:
                    print(f"Insert Error: {e}")
            conn.close()

        elapsed = time.time() - start_ts
        time.sleep(max(0, 180 - elapsed % 180))

if __name__ == "__main__":
    main()
