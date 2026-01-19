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
import sqlite3
from discordwebhook import Discord

# スクレイピング機能
from scraper import scrape_race_data, scrape_result

# ==========================================
# ⚙️ 設定エリア
# ==========================================
BET_AMOUNT = 1000
DB_FILE = "race_data.db"
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
REPORT_HOURS = [13, 18, 23]

# ★ 日本時間(JST)設定
t_delta = datetime.timedelta(hours=9)
JST = datetime.timezone(t_delta, 'JST')

# ==========================================
# 🗄️ データベース管理
# ==========================================
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history (
        race_id TEXT PRIMARY KEY,
        date TEXT,
        time TEXT,
        place TEXT,
        race_no INTEGER,
        predict_combo TEXT,
        predict_prob REAL,
        gemini_comment TEXT,
        result_combo TEXT,
        is_win INTEGER,
        payout INTEGER,
        profit INTEGER,
        status TEXT
    )''')
    conn.commit()
    conn.close()

def log_prediction_to_db(race_id, jcd, rno, date, combo, prob, comment):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        place_name = PLACE_NAMES.get(jcd, "不明")
        now_time = datetime.datetime.now(JST).strftime('%H:%M:%S')
        c.execute('''INSERT OR IGNORE INTO history 
            (race_id, date, time, place, race_no, predict_combo, predict_prob, gemini_comment, status, result_combo, is_win, payout, profit)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (race_id, date, now_time, place_name, rno, combo, float(prob), comment, "PENDING", "", 0, 0, 0))
        conn.commit()
    except Exception as e: print(f"DB Error: {e}")
    finally: conn.close()

def update_result_to_db(race_id, result_combo, payout):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        c.execute("SELECT predict_combo FROM history WHERE race_id=?", (race_id,))
        row = c.fetchone()
        if row:
            predict_combo = row[0]
            is_win = 1 if predict_combo == result_combo else 0
            profit = (payout - BET_AMOUNT) if is_win else -BET_AMOUNT
            c.execute('''UPDATE history SET result_combo=?, is_win=?, payout=?, profit=?, status=? WHERE race_id=?''',
                (result_combo, is_win, payout, profit, "FINISHED", race_id))
            conn.commit()
            return is_win, profit
    except: pass
    finally: conn.close()
    return False, 0

def get_today_summary_from_db():
    today = datetime.datetime.now(JST).strftime('%Y%m%d')
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT count(*), sum(is_win), sum(profit) FROM history WHERE date=? AND status='FINISHED'", (today,))
    total, wins, profit = c.fetchone()
    conn.close()
    return total or 0, wins or 0, profit or 0

def get_total_balance_from_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT sum(profit) FROM history WHERE status='FINISHED'")
    balance = c.fetchone()[0]
    conn.close()
    return balance or 0

# ==========================================
# 🚀 メインロジック
# ==========================================
def load_status():
    if not os.path.exists('status.json'): return {"notified": [], "last_report": ""}
    with open('status.json', 'r') as f: return json.load(f)

def save_status(status):
    with open('status.json', 'w') as f: json.dump(status, f, indent=4)

def push_data_to_github():
    try:
        subprocess.run('git config --global user.name "github-actions[bot]"', shell=True)
        subprocess.run('git config --global user.email "github-actions[bot]@users.noreply.github.com"', shell=True)
        subprocess.run(f'git add status.json {DB_FILE}', shell=True)
        subprocess.run('git pull origin main --rebase', shell=True)
        subprocess.run('git commit -m "Update Data"', shell=True)
        subprocess.run('git push origin main', shell=True)
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

def check_deadline(deadline_str, now_dt):
    """
    締切時刻を過ぎていないかチェックする関数
    True: まだ間に合う / False: もう終わってる
    """
    try:
        if not deadline_str: return False
        # 今日の日付と、スクレイピングした時刻(HH:MM)を結合
        hm = deadline_str.split(":")
        deadline_dt = now_dt.replace(hour=int(hm[0]), minute=int(hm[1]), second=0, microsecond=0)
        
        # 締切まであと「5分」を切っていたら、もうスキップする（安全策）
        limit = deadline_dt - datetime.timedelta(minutes=5)
        return now_dt < limit
    except:
        return True # 判定できない場合は念のためチェックする

def send_daily_report(current_hour):
    total, wins, today_profit = get_today_summary_from_db()
    total_balance = get_total_balance_from_db()
    if total == 0 and current_hour != 23: return

    win_rate = (wins / total * 100) if total > 0 else 0
    emoji = "🌞" if current_hour == 13 else ("🌇" if current_hour == 18 else "🌙")
    
    msg = (f"{emoji} **{current_hour}時の収支報告**\n━━━━━━━━━━━━━━\n"
           f"📅 本日戦績: {wins}勝 {total - wins}敗\n"
           f"🎯 的中率: {win_rate:.1f}%\n"
           f"💵 **本日: {'+' if today_profit > 0 else ''}{today_profit}円**\n"
           f"💰 通算: {total_balance}円\n━━━━━━━━━━━━━━")
    discord.post(content=msg)

def main():
    start_time = time.time()
    now = datetime.datetime.now(JST)
    today = now.strftime('%Y%m%d')
    current_hour = now.hour
    
    print(f"🚀 Bot起動: {now.strftime('%H:%M')} (ラウンド巡回＆時間判定)")
    
    if current_hour == 23 and now.minute > 15:
        print("💤 業務終了時刻です")
        return

    init_db()
    session = requests.Session()
    status = load_status()

    # モデル読み込み
    if not os.path.exists(MODEL_FILE):
        if os.path.exists(ZIP_MODEL):
            with zipfile.ZipFile(ZIP_MODEL, 'r') as f: f.extractall()
        elif os.path.exists('model_part_1'):
            with open(ZIP_MODEL, 'wb') as f_out:
                for i in range(1, 10):
                    p = f'model_part_{i}'
                    if os.path.exists(p): with open(p, 'rb') as f_in: f_out.write(f_in.read())
            with zipfile.ZipFile(ZIP_MODEL, 'r') as f: f.extractall()

    try: bst = lgb.Booster(model_file=MODEL_FILE)
    except: return

    # --- 1. 結果確認 (これは会場ループでOK) ---
    print("📊 結果確認中...")
    updated = False
    for item in status["notified"]:
        if item.get("checked"): continue
        if "jcd" not in item:
            try: item["date"], item["jcd"], item["rno"] = item["id"].split("_")[0], int(item["id"].split("_")[1]), int(item["id"].split("_")[2])
            except: continue

        res = scrape_result(session, item["jcd"], item["rno"], item["date"])
        if res:
            is_win, profit = update_result_to_db(item["id"], res["combo"], res["payout"])
            item["checked"] = True
            updated = True
            total_balance = get_total_balance_from_db()
            place = PLACE_NAMES.get(item["jcd"], "会場")
            discord.post(content=f"{'🎊 的中' if is_win else '💀 外れ'} {place}{item['rno']}R\n予測:{item['combo']}→結果:{res['combo']}\n収支:{'+' if profit>0 else ''}{profit}円\n通算:{total_balance}円")
    
    if updated:
        save_status(status)
        push_data_to_github()

    # --- 2. 定期報告 ---
    report_key = f"{today}_{current_hour}"
    if current_hour in REPORT_HOURS and status.get("last_report") != report_key:
        send_daily_report(current_hour)
        status["last_report"] = report_key
        save_status(status)
        push_data_to_github()

    # --- 3. 新規予想 (ここが大改革！) ---
    if current_hour < 22:
        print("🔍 ラウンド順にパトロール中...")
        
        # ★大改革: ラウンド(1R〜12R)を外側のループにする
        # これで「全会場の1R」→「全会場の2R」の順に見る
        for rno in range(1, 13):
            
            # タイムアウト対策
            if time.time() - start_time > 3000: break
            venue_updated = False
            
            for jcd in range(1, 25):
                race_id = f"{today}_{str(jcd).zfill(2)}_{rno}"
                if any(n['id'] == race_id for n in status["notified"]): continue

                try:
                    # データ取得
                    raw_data = scrape_race_data(session, jcd, rno, today)
                    if raw_data is None: continue

                    # ★時間チェック: 締切を過ぎていたらスキップ
                    deadline = raw_data.get('deadline_time')
                    if not check_deadline(deadline, now):
                        # print(f"SKIP: {jcd}場{rno}R (締切 {deadline} 経過済)")
                        continue

                    # ここまで来たら「まだ間に合うレース」
                    df = pd.DataFrame([raw_data])
                    df = engineer_features(df)
                    cols = ['jcd', 'rno', 'wind', 'wr_1_vs_avg']
                    for i in range(1, 7): cols.extend([f'wr{i}', f'st{i}', f'ex{i}', f'power_idx_{i}'])
                    for i in range(1, 6): cols.extend([f'st_gap_{i}_{i+1}', f'wr_gap_{i}_{i+1}'])

                    probs = bst.predict(df[cols])[0]
                    win_probs = calculate_tansho_probs(probs)
                    best_boat = max(win_probs, key=win_probs.get)
                    best_idx = np.argmax(probs)
                    combo, prob = COMBOS[best_idx], probs[best_idx]

                    if prob > 0.4 or win_probs[best_boat] > 0.6:
                        place = PLACE_NAMES.get(jcd, "会場")
                        try:
                            prompt = f"{place}{rno}R。単勝{best_boat}({win_probs[best_boat]:.0%})、二連単{combo}({prob:.0%})。推奨理由を一言。"
                            res_gemini = model_gemini.generate_content(prompt).text
                        except: res_gemini = "Gemini応答なし"

                        msg = (f"🚀 **勝負レース!** {place}{rno}R (締切 {deadline})\n"
                               f"🛶 単勝:{best_boat}艇({win_probs[best_boat]:.0%})\n"
                               f"🔥 二連単:{combo}({prob:.0%})\n"
                               f"🤖 {res_gemini}\n"
                               f"[出走表](https://www.boatrace.jp/owpc/pc/race/racelist?rno={rno}&jcd={jcd:02d}&hd={today})")
                        discord.post(content=msg)
                        log_prediction_to_db(race_id, jcd, rno, today, combo, prob, res_gemini)
                        status["notified"].append({"id": race_id, "jcd": jcd, "rno": rno, "date": today, "combo": combo, "checked": False})
                        venue_updated = True
                        time.sleep(1) # 連投防止
                except Exception as e:
                    # print(f"Err {race_id}: {e}")
                    continue
            
            # ラウンドごとに保存する
            if venue_updated:
                save_status(status)
                push_data_to_github()

    print("✅ 完了")

if __name__ == "__main__":
    main()
