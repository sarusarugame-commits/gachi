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
    """
    通知履歴(status.json)をGitHubに強制保存する関数
    これを行わないと、次回の起動時に記憶がリセットされて重複通知が発生する
    """
    try:
        print("💾 履歴をGitHubに保存中...")
        subprocess.run('git config --global user.name "github-actions[bot]"', shell=True)
        subprocess.run('git config --global user.email "github-actions[bot]@users.noreply.github.com"', shell=True)
        subprocess.run('git add status.json', shell=True)
        subprocess.run('git commit -m "Update status: Avoid duplicates"', shell=True)
        subprocess.run('git push', shell=True)
        print("✅ 保存完了")
    except Exception as e:
        print(f"⚠️ 保存失敗: {e}")

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
    """
    二連単の確率から単勝（1着）の確率を逆算する
    例: 1号艇の勝率 = (1-2) + (1-3) + (1-4) + (1-5) + (1-6) の確率の合計
    """
    win_probs = {i: 0.0 for i in range(1, 7)}
    for idx, combo in enumerate(COMBOS):
        first = int(combo.split('-')[0])
        win_probs[first] += probs[idx]
    return win_probs

def main():
    print("🚀 Bot起動: 単勝対応 & 重複防止版")
    session = requests.Session()
    status = load_status()
    today = datetime.datetime.now().strftime('%Y%m%d')

    # --- 1. モデル準備 ---
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

    # --- 2. 結果確認 ---
    print("📊 結果を確認中...")
    changes_made = False
    for item in status["notified"]:
        if item.get("checked"): continue
        
        if "jcd" not in item:
            try:
                parts = item["id"].split("_")
                item["date"] = parts[0]
                item["jcd"] = int(parts[1])
                item["rno"] = int(parts[2])
            except: continue

        try:
            res = scrape_result(session, item["jcd"], item["rno"], item["date"])
            if res:
                is_win = (res["combo"] == item["combo"])
                payout = res["payout"] if is_win else 0
                profit = payout - BET_AMOUNT
                status["total_balance"] += profit
                item["checked"] = True
                changes_made = True
                
                place = PLACE_NAMES.get(item["jcd"], f"{item['jcd']}場")
                discord.post(content=(
                    f"{'🎊 **的中！**' if is_win else '💀 不的中'}\n"
                    f"レース: {place} {item['rno']}R\n"
                    f"予測: {item['combo']} → 結果: {res['combo']}\n"
                    f"収支: {'+' if profit > 0 else ''}{profit}円\n"
                    f"💰 通算: {status['total_balance']}円"
                ))
        except: pass
    
    if changes_made:
        save_status(status)

    # --- 3. 新規予想 (単勝 & 二連単) ---
    print("🔍 パトロール中...")
    new_notifications = False
    
    for jcd in range(1, 25):
        for rno in range(1, 13):
            race_id = f"{today}_{str(jcd).zfill(2)}_{rno}"
            if any(n['id'] == race_id for n in status["notified"]): continue

            try:
                raw_data = scrape_race_data(session, jcd, rno, today)
                if raw_data is None: continue

                df = pd.DataFrame([raw_data])
                df = engineer_features(df)
                
                features = ['jcd', 'rno', 'wind', 'wr_1_vs_avg']
                for i in range(1, 7): features.extend([f'wr{i}', f'st{i}', f'ex{i}', f'power_idx_{i}'])
                for i in range(1, 6): features.extend([f'st_gap_{i}_{i+1}', f'wr_gap_{i}_{i+1}'])
                
                probs = bst.predict(df[features])[0]
                
                # ★単勝確率の計算
                win_probs = calculate_tansho_probs(probs)
                best_boat = max(win_probs, key=win_probs.get)
                best_win_prob = win_probs[best_boat]

                # 二連単の最有力
                best_idx = np.argmax(probs)
                combo = COMBOS[best_idx]
                prob = probs[best_idx]
                
                # 通知判定 (二連単40%超え または 単勝60%超え)
                if prob > 0.4 or best_win_prob > 0.6:
                    place_name = PLACE_NAMES.get(jcd, f"{jcd}場")
                    
                    # Geminiコメント
                    prompt = f"{place_name}{rno}R。単勝{best_boat}号艇(確率{best_win_prob:.2%})、二連単{combo}(確率{prob:.2%})。推奨理由を一言で。"
                    try:
                        res_gemini = model_gemini.generate_content(prompt).text
                    except:
                        res_gemini = "Gemini応答なし"

                    vote_url = f"https://www.boatrace.jp/owpc/pc/race/racelist?rno={rno}&jcd={jcd:02d}&hd={today}"
                    live_url = f"https://www.boatrace.jp/owpc/pc/race/live?jcd={jcd:02d}&rno={rno}"

                    msg = (
                        f"🚀 **勝負レース発見！**\n"
                        f"🏁 **{place_name} {rno}R**\n"
                        f"🛶 **単勝推奨**: **{best_boat}号艇** (確率 {best_win_prob:.0%})\n"
                        f"🔥 **2連単**: **{combo}** (確率 {prob:.0%})\n"
                        f"🤖 {res_gemini}\n\n"
                        f"🗳 [出走表]({vote_url}) | 📺 [ライブ]({live_url})"
                    )

                    discord.post(content=msg)
                    
                    status["notified"].append({
                        "id": race_id, "jcd": jcd, "rno": rno, 
                        "date": today, "combo": combo, "checked": False
                    })
                    save_status(status)
                    new_notifications = True
                
                time.sleep(0.5)
            except Exception as e:
                print(f"⚠️ Error {race_id}: {e}")

    # --- 4. 最後に必ず履歴を保存してプッシュ ---
    if new_notifications or changes_made:
        push_status_to_github()

    print("✅ 完了")

if __name__ == "__main__":
    main()
