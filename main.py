import os
import json
import datetime
import time
import pandas as pd
import numpy as np
import lightgbm as lgb
import google.generativeai as genai
import zipfile
from discordwebhook import Discord

# ★スクレイピング機能の読み込み
from scraper import scrape_race_data

# ==========================================
# ⚙️ 設定エリア
# ==========================================
BET_AMOUNT = 1000

# APIキーの確認と設定
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("❌ 【重要】GEMINI_API_KEY が設定されていません！")
else:
    genai.configure(api_key=api_key)

model_gemini = genai.GenerativeModel('gemini-3-flash-preview')
discord = Discord(url=os.environ["DISCORD_WEBHOOK_URL"])

MODEL_FILE = 'boat_model_nirentan.txt'
ZIP_MODEL = 'model.zip'
COMBOS = [f"{f}-{s}" for f in range(1, 7) for s in range(1, 7) if f != s]

# 会場名リスト (1~24)
PLACE_NAMES = {
    1: "桐生", 2: "戸田", 3: "江戸川", 4: "平和島", 5: "多摩川", 6: "浜名湖",
    7: "蒲郡", 8: "常滑", 9: "津", 10: "三国", 11: "びわこ", 12: "住之江",
    13: "尼崎", 14: "鳴門", 15: "丸亀", 16: "児島", 17: "宮島", 18: "徳山",
    19: "下関", 20: "若松", 21: "芦屋", 22: "福岡", 23: "唐津", 24: "大村"
}

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
    df['jcd'] = df['jcd'].astype('category')
    return df

def main():
    print("🚀 Bot起動: 親切通知モード (全24会場巡回)")
    
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
    except Exception as e:
        print(f"❌ モデル読み込み失敗: {e}")
        return

    status = load_status()
    today = datetime.datetime.now().strftime('%Y%m%d')
    
    # 全会場巡回
    for jcd in range(1, 25):
        for rno in range(1, 13):
            race_id = f"{today}_{str(jcd).zfill(2)}_{rno}"
            
            if any(n['id'] == race_id for n in status["notified"]):
                continue

            try:
                print(f"🔍 Checking {race_id}...")
                raw_data = scrape_race_data(None, jcd, rno, today)
                if raw_data is None: continue

                df = pd.DataFrame([raw_data])
                df = engineer_features(df)
                
                features = ['jcd', 'rno', 'wind', 'wr_1_vs_avg']
                for i in range(1, 7): features.extend([f'wr{i}', f'st{i}', f'ex{i}', f'power_idx_{i}'])
                for i in range(1, 6): features.extend([f'st_gap_{i}_{i+1}', f'wr_gap_{i}_{i+1}'])
                
                probs = bst.predict(df[features])[0]
                best_idx = np.argmax(probs)
                combo = COMBOS[best_idx]
                prob = probs[best_idx]
                
                # 閾値判定 (テスト用に0.4のままですが、本番なら上げてもOK)
                if prob > 0.4:
                    print(f"✨ 候補発見: {race_id} {combo} (確率:{prob:.2%})")
                    
                    place_name = PLACE_NAMES.get(jcd, f"{jcd}場")
                    
                    # Geminiコメント生成
                    prompt = f"競艇予測AIです。{place_name}{rno}R、的中率{prob:.2%}で「{combo}」を有力と判断しました。この買い目は推奨できますか？一言で回答して。"
                    try:
                        res = model_gemini.generate_content(prompt).text
                    except Exception as e:
                        res = f"Gemini応答失敗: {e}"

                    # URL生成
                    # 公式の出走表ページ (ここから投票ボタンにも行ける)
                    vote_url = f"https://www.boatrace.jp/owpc/pc/race/racelist?rno={rno}&jcd={jcd:02d}&hd={today}"
                    # 公式ライブ配信ページ
                    live_url = f"https://www.boatrace.jp/owpc/pc/race/live?jcd={jcd:02d}&rno={rno}"

                    # Discord通知メッセージ作成
                    message = (
                        f"🚀 **勝負レース発見！**\n"
                        f"🏁 **{place_name} {rno}R**\n"
                        f"🔥 推奨: **{combo}**\n"
                        f"📊 AI確率: {prob:.2%}\n"
                        f"🤖 Gemini: {res}\n\n"
                        f"🗳 [投票・出走表]({vote_url})\n"
                        f"📺 [ライブ配信]({live_url})"
                    )

                    discord.post(content=message)
                    
                    status["notified"].append({"id": race_id, "combo": combo})
                    save_status(status)
                
                time.sleep(1)

            except Exception as e:
                print(f"⚠️ Error {race_id}: {e}")

if __name__ == "__main__":
    main()
