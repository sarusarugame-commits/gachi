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

# スクレイピング機能
from scraper import scrape_race_data

# ==========================================
# ⚙️ 設定エリア
# ==========================================
DB_FILE = "race_data.db"

# ★閾値設定
THRESHOLD_NIRENTAN = 0.50
THRESHOLD_TANSHO   = 0.75

# ★Groq設定
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

# ==========================================
# 🤖 Groq API & Discord
# ==========================================
def call_groq_api(prompt):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key: return "APIキー未設定"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": GROQ_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    
    try:
        res = requests.post(GROQ_API_URL, headers=headers, json=data, timeout=10)
        if res.status_code == 200:
            return res.json()['choices'][0]['message']['content']
        else:
            print(f"⚠️ Groq Error: {res.status_code} {res.text}")
            return f"エラー({res.status_code})"
    except Exception as e:
        print(f"⚠️ Groq通信エラー: {e}")
        return "応答なし"

def send_discord(content):
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not url: return
    try: requests.post(url, json={"content": content}, timeout=10)
    except: pass

# ==========================================
# 🗄️ DB & Logic
# ==========================================
def init_db():
    conn = sqlite3.connect(DB_FILE, timeout=30)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history (
        race_id TEXT PRIMARY KEY, date TEXT, time TEXT, place TEXT, race_no INTEGER,
        predict_combo TEXT, predict_prob REAL, gemini_comment TEXT,
        result_combo TEXT, is_win INTEGER, payout INTEGER, profit INTEGER, status TEXT
    )''')
    conn.commit()
    conn.close()

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
        
        if d_dt < now_dt - datetime.timedelta(hours=1):
             d_dt += datetime.timedelta(days=1)
        
        if now_dt > d_dt: return False
        
        return (d_dt - now_dt) <= datetime.timedelta(minutes=60)
    except: return True

def process_prediction(jcd, today, notified_ids, bst):
    pred_list = []
    sess = requests.Session()
    now = datetime.datetime.now(JST)
    
    for rno in range(1, 13):
        rid = f"{today}_{str(jcd).zfill(2)}_{rno}"
        if rid in notified_ids: continue
        
        try:
            raw = scrape_race_data(sess, jcd, rno, today)
            if not raw: continue 
            
            if not is_target_race(raw.get('deadline_time'), now): continue
            
            df = engineer_features(pd.DataFrame([raw]))
            cols = ['jcd', 'rno', 'wind', 'wr_1_vs_avg']
            for i in range(1, 7): cols.extend([f'wr{i}', f'st{i}', f'ex{i}', f'power_idx_{i}'])
            for i in range(1, 6): cols.extend([f'st_gap_{i}_{i+1}', f'wr_gap_{i}_{i+1}'])
            
            probs = bst.predict(df[cols])[0]
            win_p = calculate_tansho(probs)
            best_b = max(win_p, key=win_p.get)
            best_idx = np.argmax(probs)
            combo, prob = COMBOS[best_idx], probs[best_idx]

            if prob >= THRESHOLD_NIRENTAN or win_p[best_b] >= THRESHOLD_TANSHO:
                place = PLACE_NAMES.get(jcd, "会場")
                prompt = f"{place}{rno}R。単勝{best_b}({win_p[best_b]:.0%})、二連単{combo}({prob:.0%})。推奨理由を一言。"
                comment = call_groq_api(prompt)
                
                pred_list.append({
                    'id': rid, 'jcd': jcd, 'rno': rno, 'date': today, 
                    'combo': combo, 'prob': prob, 'best_boat': best_b, 
                    'win_prob': win_p[best_b], 'comment': comment, 
                    'deadline': raw.get('deadline_time')
                })
        except: continue
    return pred_list

def main():
    print(f"🚀 [Main] 高速予想Bot起動 (Model: {GROQ_MODEL_NAME})")
    init_db()
    
    # ★修正箇所：分割ファイルの結合ロジックを復活
    if not os.path.exists(MODEL_FILE):
        # model.zip が無い場合、model_part_* を探して結合する
        if not os.path.exists(ZIP_MODEL):
            if os.path.exists('model_part_1') or os.path.exists('model_part_01'):
                print("📦 分割モデルを結合中...")
                with open(ZIP_MODEL, 'wb') as f_out:
                    for i in range(1, 20): # 最大20分割まで想定
                        # model_part_1 と model_part_01 の両パターンに対応
                        part_name = f'model_part_{i}'
                        if not os.path.exists(part_name):
                             part_name = f'model_part_{i:02d}'
                        
                        if os.path.exists(part_name):
                            with open(part_name, 'rb') as f_in: f_out.write(f_in.read())
                        else:
                            break # ファイルが途切れたら終了

        # model.zip があれば解凍
        if os.path.exists(ZIP_MODEL):
            print("📦 モデルを解凍中...")
            with zipfile.ZipFile(ZIP_MODEL, 'r') as f: f.extractall()
            print("✅ 解凍完了")
    
    try: 
        bst = lgb.Booster(model_file=MODEL_FILE)
        print("✅ モデル読み込み成功")
    except Exception as e:
        print(f"🔥 モデル読み込み失敗: {e}")
        # GitHub Actionsでのデバッグ用にファイル一覧を表示
        print("📂 現在のフォルダ構成:")
        print(os.listdir('.'))
        return

    while True:
        start_ts = time.time()
        now = datetime.datetime.now(JST)
        today = now.strftime('%Y%m%d')
        
        # 23:10 終了
        if now.hour >= 23 and now.minute >= 10:
            print("🌙 業務終了")
            break

        conn = sqlite3.connect(DB_FILE, timeout=30)
        c = conn.cursor()
        c.execute("SELECT race_id FROM history")
        notified_ids = set(row[0] for row in c.fetchall())
        conn.close()

        print(f"⚡️ スキャン: {now.strftime('%H:%M:%S')}")
        
        new_preds = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            futures = [executor.submit(process_prediction, jcd, today, notified_ids, bst) for jcd in range(1, 25)]
            for f in concurrent.futures.as_completed(futures):
                try: new_preds.extend(f.result())
                except: pass
        
        if new_preds:
            conn = sqlite3.connect(DB_FILE, timeout=30)
            c = conn.cursor()
            for pred in new_preds:
                now_str = datetime.datetime.now(JST).strftime('%H:%M:%S')
                place = PLACE_NAMES.get(pred['jcd'], "不明")
                c.execute("INSERT OR IGNORE INTO history VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (pred['id'], pred['date'], now_str, place, pred['rno'], pred['combo'], float(pred['prob']), pred['comment'], "PENDING", "", 0, 0, 0))
                
                t_disp = f"(締切 {pred['deadline']})" if pred['deadline'] else ""
                msg = (f"🔥 **勝負レース予想** {place}{pred['rno']}R {t_disp}\n"
                       f"🛶 単勝:{pred['best_boat']}艇({pred['win_prob']:.0%})\n"
                       f"🎯 二連単:{pred['combo']}({pred['prob']:.0%})\n"
                       f"🤖 {pred['comment']}\n"
                       f"[出走表](https://www.boatrace.jp/owpc/pc/race/racelist?rno={pred['rno']}&jcd={pred['jcd']:02d}&hd={pred['date']})")
                send_discord(msg)
                print(f"✅ 通知: {place}{pred['rno']}R")
            conn.commit()
            conn.close()

        elapsed = time.time() - start_ts
        sleep_time = max(0, 180 - elapsed)
        print(f"⏳ 待機: {int(sleep_time)}秒")
        time.sleep(sleep_time)

if __name__ == "__main__":
    main()
