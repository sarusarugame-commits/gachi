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
THRESHOLD_NIRENTAN = 0.50
THRESHOLD_TANSHO   = 0.75

# 毎時報告 (8時〜23時)
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

# ==========================================
# 🛠️ ユーティリティ & API
# ==========================================
def extract_odds_value(odds_text, target_boat=None):
    try:
        # "1.5" のような単数値の場合
        if re.match(r"^\d+\.\d+$", odds_text):
            return float(odds_text)
        
        # 文字列の中から数値を探す
        match = re.search(r"(\d+\.\d+)", str(odds_text))
        if match:
            return float(match.group(1))
    except: pass
    return 0.0

def call_groq_api(prompt):
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key: return "APIキー未設定"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": GROQ_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3 # 論理的判断のため温度を下げる
    }
    try:
        res = requests.post(GROQ_API_URL, headers=headers, json=data, timeout=30)
        if res.status_code == 200:
            return res.json()['choices'][0]['message']['content']
        else:
            print(f"⚠️ [Groq] Error: {res.status_code}")
            return f"エラー({res.status_code})"
    except: return "応答なし"

def send_discord(content):
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not url: return
    try: requests.post(url, json={"content": content}, timeout=10)
    except: pass

def init_db():
    conn = sqlite3.connect(DB_FILE, timeout=30)
    c = conn.cursor()
    # ★重要: WALモードを有効化して、読み書きの競合を防ぐ
    c.execute("PRAGMA journal_mode=WAL;")
    c.execute('''CREATE TABLE IF NOT EXISTS history (
        race_id TEXT PRIMARY KEY, date TEXT, time TEXT, place TEXT, race_no INTEGER,
        predict_combo TEXT, predict_prob REAL, gemini_comment TEXT,
        result_combo TEXT, is_win INTEGER, payout INTEGER, profit INTEGER, status TEXT
    )''')
    conn.commit()
    conn.close()

# ==========================================
# 📊 報告専用スレッド
# ==========================================
def report_worker():
    print("📋 [Report] 報告スレッド起動")
    last_report_key = ""
    
    while True:
        try:
            # DB接続
            conn = sqlite3.connect(DB_FILE, timeout=30)
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            
            # 1. 結果チェック (PENDINGのものを確認)
            c.execute("SELECT * FROM history WHERE status='PENDING'")
            pending_races = c.fetchall()
            
            sess = requests.Session()
            updates = 0
            
            for race in pending_races:
                try:
                    parts = race['race_id'].split('_')
                    date_str, jcd, rno = parts[0], int(parts[1]), int(parts[2])
                    
                    res = scrape_result(sess, jcd, rno, date_str)
                    if res:
                        is_win = 1 if race['predict_combo'] == res['combo'] else 0
                        profit = (res['payout'] - BET_AMOUNT) if is_win else -BET_AMOUNT
                        
                        # DB更新 (このスレッド内で完結させる)
                        c.execute("""
                            UPDATE history 
                            SET result_combo=?, is_win=?, payout=?, profit=?, status='FINISHED' 
                            WHERE race_id=?
                        """, (res['combo'], is_win, res['payout'], profit, race['race_id']))
                        
                        updates += 1
                        
                        place = PLACE_NAMES.get(jcd, "会場")
                        msg = (f"{'🎊 的中' if is_win else '💀 外れ'} {place}{rno}R\n"
                               f"予測:{race['predict_combo']} → 結果:{res['combo']}\n"
                               f"収支:{'+' if profit>0 else ''}{profit}円")
                        send_discord(msg)
                        print(f"📊 [Report] 結果判明: {place}{rno}R")
                        time.sleep(1)
                except: continue
            
            if updates > 0:
                conn.commit()

            # 2. 定期報告チェック
            now = datetime.datetime.now(JST)
            today = now.strftime('%Y%m%d')
            current_key = f"{today}_{now.hour}"
            
            if now.hour in REPORT_HOURS and last_report_key != current_key:
                c.execute("SELECT count(*), sum(is_win), sum(profit) FROM history WHERE date=? AND status='FINISHED'", (today,))
                cnt, wins, profit = c.fetchone()
                c.execute("SELECT count(*) FROM history WHERE date=? AND status='PENDING'", (today,))
                pending_cnt = c.fetchone()[0]
                
                # 生存報告も兼ねて必ず送る
                status_emoji = "🟢" if (pending_cnt > 0) else "💤"
                msg = (f"**🛠️ {now.hour}時の定期報告**\n"
                       f"状態: {status_emoji} 稼働中\n"
                       f"✅ 完了: {cnt or 0}R (的中: {wins or 0})\n"
                       f"⏳ 待機: {pending_cnt or 0}R\n"
                       f"💵 収支: {'+' if (profit or 0)>0 else ''}{profit or 0}円")
                
                send_discord(msg)
                print(f"📢 [Report] 定期報告送信: {now.hour}時")
                last_report_key = current_key

            conn.close()

        except Exception as e:
            print(f"🔥 [Report] Error: {e}")
            traceback.print_exc()
        
        # 5分待機
        time.sleep(300)

# ==========================================
# 🚤 予想ロジック (メインスレッド)
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

# ★修正: オッズ取得のリトライ機能
def get_odds_with_retry(sess, jcd, rno, today, best_b, combo):
    for _ in range(3): # 3回リトライ
        odds_data = scrape_odds(sess, jcd, rno, today, target_boat=str(best_b), target_combo=combo)
        if odds_data['tansho'] != "---" or odds_data['nirentan'] != "---":
            return odds_data
        time.sleep(2)
    return {"tansho": "1.0", "nirentan": "1.0"} # 失敗時は最低値

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
                print(f"🎯 [Main] 候補発見: {place}{rno}R (Model:{win_p[best_b]:.0%}) -> オッズ確認")
                
                # オッズ取得（リトライ付き）
                odds_data = get_odds_with_retry(sess, jcd, rno, today, best_b, combo)
                
                # 期待値計算 (AI単勝勝率 × リアルタイム単勝オッズ)
                real_odds = extract_odds_value(odds_data['tansho'])
                if real_odds == 0: real_odds = 1.0
                expected_value = real_odds * win_p[best_b]
                
                print(f"💰 [Main] 期待値: {expected_value:.2f} (オッズ{real_odds} x 勝率{win_p[best_b]:.2f})")

                # ★修正: AIへの指示を数値ベースで厳格化
                prompt = f"""
                あなたはボートレースの投資アルゴリズムです。以下の数値に基づき判断してください。
                
                【データ】
                ・算出された期待値(EV): {expected_value:.2f}
                ・基準: EVが1.0以上なら「利益見込みあり」、1.0未満なら「期待値不足」
                
                【指示】
                1. 結論は必ず「買い」または「見（ケン）」のどちらかで始めてください。
                2. EVが1.0未満の場合は、必ず「見」としてください。
                3. EVが1.0以上の場合は「買い」としてください。
                4. 理由は40文字以内で簡潔に述べてください。
                
                例：「見。期待値0.85でリスクに見合わず。」
                例：「買い。期待値1.20でプラス収支が狙える。」
                """
                
                comment = call_groq_api(prompt)
                
                pred_list.append({
                    'id': rid, 'jcd': jcd, 'rno': rno, 'date': today, 
                    'combo': combo, 'prob': prob, 'best_boat': best_b, 
                    'win_prob': win_p[best_b], 'comment': comment, 
                    'deadline': raw.get('deadline_time'),
                    'odds': odds_data,
                    'ev': expected_value
                })
        except: continue
    return pred_list

def main():
    print(f"🚀 [Main] 完全統合Bot起動 (Model: {GROQ_MODEL_NAME})")
    init_db()
    
    if not os.path.exists(MODEL_FILE):
        if not os.path.exists(ZIP_MODEL):
            if os.path.exists('model_part_1') or os.path.exists('model_part_01'):
                print("📦 分割モデルを結合中...")
                with open(ZIP_MODEL, 'wb') as f_out:
                    for i in range(1, 20):
                        part_name = f'model_part_{i}'
                        if not os.path.exists(part_name): part_name = f'model_part_{i:02d}'
                        if os.path.exists(part_name):
                            with open(part_name, 'rb') as f_in: f_out.write(f_in.read())
                        else: break
        if os.path.exists(ZIP_MODEL):
            print("📦 モデルを解凍中...")
            with zipfile.ZipFile(ZIP_MODEL, 'r') as f: f.extractall()
    
    try: bst = lgb.Booster(model_file=MODEL_FILE)
    except Exception as e:
        print(f"🔥 モデル読み込み失敗: {e}")
        return

    t = threading.Thread(target=report_worker, daemon=True)
    t.start()

    start_ts = time.time()

    while True:
        now = datetime.datetime.now(JST)
        today = now.strftime('%Y%m%d')
        
        if now.hour >= 23 and now.minute >= 10:
            print("🌙 業務終了 (23:10)")
            break

        if time.time() - start_ts > 21000:
            print("🛑 タイムリミット (再起動待機)")
            break

        conn = sqlite3.connect(DB_FILE, timeout=30)
        c = conn.cursor()
        c.execute("SELECT race_id FROM history")
        notified_ids = set(row[0] for row in c.fetchall())
        conn.close()

        print(f"⚡️ [Main] スキャン: {now.strftime('%H:%M:%S')}")
        
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
                odds_url = f"https://www.boatrace.jp/owpc/pc/race/oddstf?rno={pred['rno']}&jcd={pred['jcd']:02d}&hd={pred['date']}"
                odds_t = pred['odds'].get('tansho', '-')
                odds_n = pred['odds'].get('nirentan', '-')
                ev_val = pred.get('ev', 0.0)

                msg = (f"🔥 **{place}{pred['rno']}R** {t_disp}\n"
                       f"🛶 本命: {pred['best_boat']}号艇 (勝率:{pred['win_prob']:.0%})\n"
                       f"🎯 推奨: {pred['combo']} (的中:{pred['prob']:.0%})\n"
                       f"💰 オッズ: 単{odds_t} / 2単{odds_n}\n"
                       f"📈 期待値: {ev_val:.2f} (1.0超で狙い目)\n"
                       f"━━━━━━━━━━━━━━\n"
                       f"🤖 **{pred['comment']}**\n"
                       f"━━━━━━━━━━━━━━\n"
                       f"📊 [オッズ確認]({odds_url})")
                send_discord(msg)
                print(f"✅ [Main] 通知: {place}{pred['rno']}R")
            conn.commit()
            conn.close()

        elapsed = time.time() - start_ts
        sleep_time = max(0, 180 - elapsed % 180)
        print(f"⏳ [Main] 待機: {int(sleep_time)}秒")
        time.sleep(sleep_time)

if __name__ == "__main__":
    main()
