import os
import json
import datetime
import time
import pandas as pd
import numpy as np
import lightgbm as lgb
import google.generativeai as genai
import zipfile
import requests
import subprocess
from discordwebhook import Discord

# スクレイピング機能の読み込み
from scraper import scrape_race_data, scrape_result

# ==========================================
# ⚙️ 設定エリア
# ==========================================
BET_AMOUNT = 1000
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model_gemini = genai.GenerativeModel('gemini-1.5-flash')
discord = Discord(url=os.environ["DISCORD_WEBHOOK_URL"])

MODEL_FILE = 'boat_model_nirentan.txt'
ZIP_MODEL = 'model.zip'
COMBOS = [f"{f}-{s}" for f in range(1, 7) for s in range(1, 7) if f != s]
PLACE_NAMES = {
    1: "桐生", 2: "戸田", 3: "江戸川", 4: "平和島", 5: "多摩川", 6: "浜名湖",
    7: "蒲郡", 8: "常滑", 9: "津", 10: "三国", 11: "びわこ", 12: "住之江",
    13: "尼崎", 14: "鳴門", 15: "丸亀", 16: "児島", 17: "宮島", 18: "徳山",
    19: "下関", 20: "若松", 21: "芦屋", 22: "福岡", 23: "唐津", 24: "大村"
}

def load_status():
    if not os.path.exists('status.json'):
        return {"notified": [], "total_balance": 0}
    with open('status.json', 'r') as f:
        return json.load(f)

def save_status(status):
    with open('status.json', 'w') as f:
        json.dump(status, f, indent=4)

def push_status_to_github():
    """履歴を保存して重複とタイムアウト後の二重送りを防ぐ"""
    try:
        subprocess.run('git config --global user.name "github-actions[bot]"', shell=True)
        subprocess.run('git config --global user.email "github-actions[bot]@users.noreply.github.com"', shell=True)
        subprocess.run('git add status.json', shell=True)
        # 競合回避のため、一度最新を取り込んでからpush
        subprocess.run('git pull origin main --rebase', shell=True)
        subprocess.run('git commit -m "Update status: Progress saved"', shell=True)
        subprocess.run('git push origin main', shell=True)
        print("💾 進捗をGitHubに保存しました")
    except: pass

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

def calculate_tansho_probs(probs):
    win_probs = {i: 0.0 for i in range(1, 7)}
    for idx, combo in enumerate(COMBOS):
        first = int(combo.split('-')[0])
        win_probs[first] += probs[idx]
    return win_probs

def main():
    start_time = time.time()
    print("🚀 Bot起動: 高速＆レジュームモード")
    session = requests.Session()
    status = load_status()
    today = datetime.datetime.now().strftime('%Y%m%d')

    # モデル準備
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
    except: return

    # 1. 結果確認 (未確認のものだけ)
    print("📊 結果確認中...")
    updated = False
    for item in status["notified"]:
        if item.get("checked") or item.get("date") != today: continue
        res = scrape_result(session, item["jcd"], item["rno"], item["date"])
        if res:
            is_win = (res["combo"] == item["combo"])
            status["total_balance"] += (res["payout"] - BET_AMOUNT) if is_win else -BET_AMOUNT
            item["checked"] = True
            updated = True
            place = PLACE_NAMES.get(item["jcd"], "会場")
            discord.post(content=f"{'🎊 的中' if is_win else '💀 外れ'} {place}{item['rno']}R\n予測:{item['combo']}→結果:{res['combo']}\n通算:{status['total_balance']}円")
    if updated: save_status(status)

    # 2. 新規予想
    print("🔍 パトロール中...")
    for jcd in range(1, 25):
        # タイムアウト対策：実行時間が50分を超えたら安全に終了して進捗保存
        if time.time() - start_time > 3000:
            print("⏳ タイムアウト防止のため一旦終了します")
            break
            
        venue_updated = False
        for rno in range(1, 13):
            race_id = f"{today}_{str(jcd).zfill(2)}_{rno}"
            if any(n['id'] == race_id for n in status["notified"]): continue

            try:
                raw_data = scrape_race_data(session, jcd, rno, today)
                if raw_data is None: continue

                df = pd.DataFrame([raw_data])
                df = engineer_features(df)
                probs = bst.predict(df[['jcd', 'rno', 'wind', 'wr_1_vs_avg'] + [f'wr{i}' for i in range(1,7)] + [f'st{i}' for i in range(1,7)] + [f'ex{i}' for i in range(1,7)] + [f'power_idx_{i}' for i in range(1,7)] + [f'st_gap_{i}_{i+1}' for i in range(1,6)] + [f'wr_gap_{i}_{i+1}' for i in range(1,6)]])[0]
                
                win_probs = calculate_tansho_probs(probs)
                best_boat = max(win_probs, key=win_probs.get)
                best_idx = np.argmax(probs)
                combo, prob = COMBOS[best_idx], probs[best_idx]
                
                if prob > 0.4 or win_probs[best_boat] > 0.6:
                    place = PLACE_NAMES.get(jcd, "会場")
                    try:
                        res_gemini = model_gemini.generate_content(f"{place}{rno}R。単勝{best_boat}({win_probs[best_boat]:.0%})、二連単{combo}({prob:.0%})。推奨理由を一言。").text
                    except: res_gemini = "Gemini応答なし"

                    discord.post(content=f"🚀 **勝負レース!** {place}{rno}R\n🛶 単勝:{best_boat}艇({win_probs[best_boat]:.0%})\n🔥 二連単:{combo}({prob:.0%})\n🤖 {res_gemini}\n[出走表](https://www.boatrace.jp/owpc/pc/race/racelist?rno={rno}&jcd={jcd:02d}&hd={today})")
                    status["notified"].append({"id": race_id, "jcd": jcd, "rno": rno, "date": today, "combo": combo, "checked": False})
                    venue_updated = True
            except: continue
        
        if venue_updated:
            save_status(status)
            push_status_to_github() # 会場ごとに進捗保存

    save_status(status)
    print("✅ 巡回終了")

if __name__ == "__main__":
    main()
