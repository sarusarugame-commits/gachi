import os
import json
import datetime
import pandas as pd
import numpy as np
import lightgbm as lgb
import google.generativeai as genai
import zipfile
from discordwebhook import Discord

# ★ここが重要：新しいファイルを読み込む
from scraper import scrape_race_data

# 設定
BET_AMOUNT = 1000
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model_gemini = genai.GenerativeModel('gemini-1.5-flash')
discord = Discord(url=os.environ["DISCORD_WEBHOOK_URL"])
MODEL_FILE = 'boat_model_nirentan.txt'
ZIP_MODEL = 'model.zip'
COMBOS = [f"{f}-{s}" for f in range(1, 7) for s in range(1, 7) if f != s]

def load_status():
    if not os.path.exists('status.json'): return {"notified": []}
    with open('status.json', 'r') as f: return json.load(f)

def save_status(status):
    with open('status.json', 'w') as f: json.dump(status, f, indent=4)

def engineer_features(df):
    for i in range(1, 7):
        df[f'power_idx_{i}'] = df[f'wr{i}'] * (1.0 / (df[f'st{i}'] + 0.01))
    for i in range(1, 6):
        df[f'st_gap_{i}_{i+1}'] = df[f'st{i+1}'] - df[f'st{i}']
        df[f'wr_gap_{i}_{i+1}'] = df[f'wr{i}'] - df[f'wr{i+1}']
    avg_wr = df[[f'wr{i}' for i in range(1, 7)]].mean(axis=1)
    df['wr_1_vs_avg'] = df['wr1'] / (avg_wr + 0.001)
    df['jcd'] = df['jcd'].astype('int')
    return df

def main():
    print("🚀 Bot起動: テストモード (1会場のみチェック)", flush=True)
    
    # 解凍処理
    if not os.path.exists(MODEL_FILE):
        if os.path.exists(ZIP_MODEL):
            with zipfile.ZipFile(ZIP_MODEL, 'r') as f: f.extractall()
        elif os.path.exists('model_part_1'):
            with open(ZIP_MODEL, 'wb') as f_out:
                for i in range(1, 10):
                    p = f'model_part_{i}'
                    if os.path.exists(p):
                        with open(p, 'rb') as f_in: f_out.write(f_in.read())
            with zipfile.ZipFile(ZIP_MODEL, 'r') as f: f.extractall()

    try:
        bst = lgb.Booster(model_file=MODEL_FILE)
    except:
        print("❌ モデル読み込み失敗", flush=True)
        return

    status = load_status()
    today = datetime.datetime.now().strftime('%Y%m%d')
    
    # ★テスト用に「1〜2 (つまり1場のみ)」に変更
    for jcd in range(1, 2): 
        for rno in range(1, 13):
            race_id = f"{today}_{str(jcd).zfill(2)}_{rno}"
            if any(n['id'] == race_id for n in status["notified"]): continue

            try:
                # スクレイピング
                print(f"🔍 Checking {race_id}...", flush=True)
                raw_data = scrape_race_data(None, jcd, rno, today)
                if raw_data is None: continue

                # 予測
                df = pd.DataFrame([raw_data])
                df = engineer_features(df)
                
                features = ['jcd', 'rno', 'wind', 'wr_1_vs_avg']
                for i in range(1, 7): features.extend([f'wr{i}', f'st{i}', f'ex{i}', f'power_idx_{i}'])
                for i in range(1, 6): features.extend([f'st_gap_{i}_{i+1}', f'wr_gap_{i}_{i+1}'])
                
                probs = bst.predict(df[features])[0]
                best_idx = np.argmax(probs)
                combo = COMBOS[best_idx]
                prob = probs[best_idx]
                
                # Gemini判定へ
                if prob > 0.4:
                    prompt = f"的中率{prob:.2f}の{combo}は買いですか？"
                    res = model_gemini.generate_content(prompt).text
                    print(f"✨ 候補発見: {race_id} {combo}", flush=True)
                    discord.post(content=f"🚀 {jcd}場{rno}R {combo}\n{res}")
                    status["notified"].append({"id": race_id, "combo": combo})
                    save_status(status)

            except Exception as e:
                print(f"⚠️ Error {race_id}: {e}", flush=True)

if __name__ == "__main__":
    main()
