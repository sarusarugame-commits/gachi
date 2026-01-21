import os
import json
import datetime
import time
import pandas as pd
import numpy as np
import lightgbm as lgb
import requests
import subprocess
import sqlite3
import concurrent.futures
import zipfile
import traceback

# スクレイピング機能
from scraper import scrape_race_data, scrape_result

# ==========================================
# ⚙️ 設定エリア
# ==========================================
BET_AMOUNT = 1000
DB_FILE = "race_data.db"
REPORT_HOURS = [13, 18, 23]

# ★閾値設定（お好みの値でOK）
THRESHOLD_NIRENTAN = 0.50
THRESHOLD_TANSHO   = 0.75

# ★モデル名修正: gemini-3 は存在しないため修正
GEMINI_MODEL_NAME = "gemini-3-flash-preview" 

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
# 🤖 Gemini API
# ==========================================
def call_gemini_api(prompt):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key: return "APIキー未設定"
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    
    try:
        res = requests.post(url, headers=headers, json=data, timeout=10)
        if res.status_code == 200:
            return res.json()['candidates'][0]['content']['parts'][0]['text']
        else:
            print(f"⚠️ Gemini Error: {res.status_code} {res.text}")
            return f"エラー({res.status_code})"
    except Exception as e:
        print(f"⚠️ Gemini Exception: {e}")
        return "応答なし"

def send_discord(content):
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not url:
        print("⚠️ DISCORD_WEBHOOK_URL未設定")
        return
    try:
        requests.post(url, json={"content": content}, timeout=10)
    except Exception as e:
        print(f"❌ Discord送信エラー: {e}")

# ==========================================
# 🗄️ DB & Git
# ==========================================
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS history (
        race_id TEXT PRIMARY KEY, date TEXT, time TEXT, place TEXT, race_no INTEGER,
        predict_combo TEXT, predict_prob REAL, gemini_comment TEXT,
        result_combo TEXT, is_win INTEGER, payout INTEGER, profit INTEGER, status TEXT
    )''')
    conn.commit()
    conn.close()

def save_and_notify(new_predictions, updated_results):
    if not new_predictions and not updated_results: return
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        # 結果更新
        for res in updated_results:
            is_win = 1 if res['predict_combo'] == res['result_combo'] else 0
            profit = (res['payout'] - BET_AMOUNT) if is_win else -BET_AMOUNT
            c.execute("UPDATE history SET result_combo=?, is_win=?, payout=?, profit=?, status=? WHERE race_id=?",
                (res['result_combo'], is_win, res['payout'], profit, "FINISHED", res['race_id']))
            place = PLACE_NAMES.get(res['jcd'], "会場")
            
            send_discord(f"{'🎊 的中' if is_win else '💀 外れ'} {place}{res['rno']}R\n予測:{res['predict_combo']}→結果:{res['result_combo']}\n収支:{'+' if profit>0 else ''}{profit}円")

        # 新規予想
        for pred in new_predictions:
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
            print(f"✅ 送信: {place}{pred['rno']}R (Prob:{pred['prob']:.2f})")
            
        conn.commit()
    except Exception as e:
        print(f"DB Error: {e}")
        traceback.print_exc()
    finally: conn.close()

def push_data():
    try:
        subprocess.run('git config --global user.name "github-actions[bot]"', shell=True)
        subprocess.run('git config --global user.email "bot@noreply.github.com"', shell=True)
        subprocess.run(f'git add status.json {DB_FILE}', shell=True)
        subprocess.run('git commit -m "Update"', shell=True)
        subprocess.run('git pull origin main --rebase', shell=True)
        subprocess.run('git push origin main', shell=True)
    except: pass

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

# ★重要：40分以内判定ロジック（安全装置付き）
def is_target_race(deadline_str, now_dt):
    try:
        # 時刻取得失敗(None) または 解析失敗時の仮値("23:59") の場合、
        # 「通知されないリスク」を避けるため、とりあえずTrue（対象）として扱う
        if not deadline_str or deadline_str == "23:59":
            return True

        hm = deadline_str.split(":")
        d_dt = now_dt.replace(hour=int(hm[0]), minute=int(hm[1]), second=0)
        
        # 日付またぎ対応
        if d_dt < now_dt - datetime.timedelta(hours=1):
             d_dt += datetime.timedelta(days=1)
        
        # 締切時刻を過ぎていたら除外
        if now_dt > d_dt: return False
        
        # 「あと40分以内」なら対象
        return (d_dt - now_dt) <= datetime.timedelta(minutes=40)
        
    except:
        # エラー発生時は念のため対象にする
        return True

def process_venue(jcd, today, notified, bst):
    res_list, pred_list = [], []
    sess = requests.Session()
    
    # 1. 結果確認
    pending_items = [i for i in notified if i['jcd'] == jcd and not i['checked']]
    for item in pending_items:
        try:
            r = scrape_result(sess, item["jcd"], item["rno"], item["date"])
            if r:
                item['checked'] = True
                res_list.append({'race_id': item['id'], 'jcd': item['jcd'], 'rno': item['rno'], 
                                 'predict_combo': item['combo'], 'result_combo': r['combo'], 'payout': r['payout']})
        except: pass

    # 2. 新規予想
    now = datetime.datetime.now(JST)
    for rno in range(1, 13):
        rid = f"{today}_{str(jcd).zfill(2)}_{rno}"
        if any(n['id'] == rid for n in notified): continue
        
        try:
            # データ取得
            raw = scrape_race_data(sess, jcd, rno, today)
            if not raw: continue 
            
            # ★ここで「40分以内」を判定
            if not is_target_race(raw.get('deadline_time'), now):
                continue
            
            # 特徴量エンジニアリング
            df = engineer_features(pd.DataFrame([raw]))
            cols = ['jcd', 'rno', 'wind', 'wr_1_vs_avg']
            for i in range(1, 7): cols.extend([f'wr{i}', f'st{i}', f'ex{i}', f'power_idx_{i}'])
            for i in range(1, 6): cols.extend([f'st_gap_{i}_{i+1}', f'wr_gap_{i}_{i+1}'])
            
            # 予測
            probs = bst.predict(df[cols])[0]
            win_p = calculate_tansho(probs)
            best_b = max(win_p, key=win_p.get)
            best_idx = np.argmax(probs)
            combo, prob = COMBOS[best_idx], probs[best_idx]

            if prob >= THRESHOLD_NIRENTAN or win_p[best_b] >= THRESHOLD_TANSHO:
                place = PLACE_NAMES.get(jcd, "会場")
                prompt = f"{place}{rno}R。単勝{best_b}({win_p[best_b]:.0%})、二連単{combo}({prob:.0%})。推奨理由を一言。"
                comment = call_gemini_api(prompt)
                
                pred_list.append({
                    'id': rid, 'jcd': jcd, 'rno': rno, 'date': today, 
                    'combo': combo, 'prob': prob, 'best_boat': best_b, 
                    'win_prob': win_p[best_b], 'comment': comment, 
                    'deadline': raw.get('deadline_time')
                })
        except Exception as e:
            continue
            
    return res_list, pred_list

def main():
    start_time = time.time()
    MAX_RUNTIME = 6 * 3600
    
    print("🚀 Bot起動 (40分以内限定モード)")
    init_db()
    
    # モデル解凍
    if not os.path.exists(MODEL_FILE):
        if os.path.exists(ZIP_MODEL):
            with zipfile.ZipFile(ZIP_MODEL, 'r') as f: f.extractall()
        elif os.path.exists('model_part_1'):
            with open(ZIP_MODEL, 'wb') as f_out:
                for i in range(1, 10):
                    if os.path.exists(f'model_part_{i}'):
                        with open(f'model_part_{i}', 'rb') as f_in: f_out.write(f_in.read())
            with zipfile.ZipFile(ZIP_MODEL, 'r') as f: f.extractall()
    
    try: 
        bst = lgb.Booster(model_file=MODEL_FILE)
    except Exception as e:
        print(f"🔥 モデル読み込み失敗: {e}")
        return

    # ★ ループ開始 ★
    while True:
        cycle_start = time.time()
        now = datetime.datetime.now(JST)
        today = now.strftime('%Y%m%d')
        
       if now.hour >= 23 and now.minute >= 10:
            print("🌙 業務終了")
            break

        if time.time() - start_time > MAX_RUNTIME - 180:
            print("💤 再起動待機")
            break
        
        if not os.path.exists('status.json'): status = {"notified": [], "last_report": ""}
        else:
            with open('status.json', 'r') as f: status = json.load(f)

        print(f"⚡️ スキャン開始: {now.strftime('%H:%M')}")
        
        # 並列処理
        all_res, all_pred = [], []
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            futures = [executor.submit(process_venue, jcd, today, status["notified"], bst) for jcd in range(1, 25)]
            for f in concurrent.futures.as_completed(futures):
                try:
                    r, p = f.result()
                    all_res.extend(r)
                    all_pred.extend(p)
                except: pass
        
        save_and_notify(all_pred, all_res)

        updated = False
        for r in all_res:
            for item in status["notified"]:
                if item['id'] == r['race_id']:
                    item['checked'] = True
                    updated = True
        for p in all_pred:
            status["notified"].append({"id": p['id'], "jcd": p['jcd'], "rno": p['rno'], 
                                       "date": p['date'], "combo": p['combo'], "checked": False})
            updated = True
        
        # 定期報告
        report_key = f"{today}_{now.hour}"
        if now.hour in REPORT_HOURS and status.get("last_report") != report_key:
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute("SELECT count(*), sum(is_win), sum(profit) FROM history WHERE date=? AND status='FINISHED'", (today,))
            cnt, wins, profit = c.fetchone()
            c.execute("SELECT count(*) FROM history WHERE date=? AND status='PENDING'", (today,))
            pending_cnt = c.fetchone()[0]
            conn.close()
            
            if cnt > 0 or pending_cnt > 0:
                msg = (f"**{now.hour}時の報告**\n"
                       f"✅ 完了:{cnt}R (勝:{wins})\n"
                       f"⏳ 待機:{pending_cnt}R\n"
                       f"💵 収支:{'+' if (profit or 0)>0 else ''}{profit or 0}円")
                send_discord(msg)
                status["last_report"] = report_key
                updated = True

        if updated:
            with open('status.json', 'w') as f: json.dump(status, f, indent=4)
            push_data()

        elapsed = time.time() - cycle_start
        sleep_time = max(0, 600 - elapsed) # 10分おき
        print(f"⏳ 待機: {int(sleep_time)}秒")
        time.sleep(sleep_time)

if __name__ == "__main__":
    main()
