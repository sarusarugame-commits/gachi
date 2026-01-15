import os, zipfile
if not os.path.exists('boat_model_nirentan.txt'):
    print('🧩 分割されたモデルを結合中...')
    with open('recombined_model.zip', 'wb') as f_out:
        for i in range(1, 10):
            part = f'model_part_{i}'
            if os.path.exists(part):
                with open(part, 'rb') as f_in: f_out.write(f_in.read())
    with zipfile.ZipFile('recombined_model.zip', 'r') as f: f.extractall()

import zipfile, os
if os.path.exists('model.zip') and not os.path.exists('boat_model_nirentan.txt'):
    with zipfile.ZipFile('model.zip', 'r') as f: f.extractall()

import os
import json
import datetime
import time
import random
import re
import requests
import pandas as pd
import numpy as np
import lightgbm as lgb
import google.generativeai as genai
from bs4 import BeautifulSoup
from discordwebhook import Discord

# ==========================================
# ⚙️ 基本設定
# ==========================================
BET_AMOUNT = 1000
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model_gemini = genai.GenerativeModel('gemini-3-flash-preview')
discord = Discord(url=os.environ["DISCORD_WEBHOOK_URL"])

# モデルとデータの定義
MODEL_FILE = 'boat_model_nirentan.txt'
COMBOS = [f"{f}-{s}" for f in range(1, 7) for s in range(1, 7) if f != s]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Referer": "https://www.boatrace.jp/",
}

# ==========================================
# 🛠️ スクレイピング・ユーティリティ
# ==========================================

def get_soup(url):
    time.sleep(random.uniform(1.5, 3.0))
    res = requests.get(url, headers=HEADERS, timeout=20)
    res.encoding = res.apparent_encoding
    return BeautifulSoup(res.text, 'html.parser')

def fetch_active_races(date):
    """今日開催されている会場とレースを取得"""
    url = f"https://www.boatrace.jp/owpc/pc/race/index?hd={date}"
    soup = get_soup(url)
    found = []
    # 開催場のリンクからjcdを取得
    for a in soup.select('a[href*="jcd="]'):
        m = re.search(r'jcd=(\d{2})', a.get('href'))
        if m: found.append(m.group(1))
    return sorted(list(set(found)))

def scrape_race_data(jcd, rno, date):
    """出走表(wr, mo, f, st)と直前情報(ex, wind)を取得"""
    data = {'jcd': int(jcd), 'rno': int(rno)}
    
    # 1. 出走表ページ
    url_prog = f"https://www.boatrace.jp/owpc/pc/race/racelist?rno={rno}&jcd={jcd}&hd={date}"
    soup_prog = get_soup(url_prog)
    
    # 選手の勝率(wr), モーター(mo), F数(f), 平均ST(st)
    # boatrace.jpの出走表テーブル構造を解析
    for i in range(1, 7):
        tbody = soup_prog.select(f'tbody.is-fs12')[i-1]
        # 全国勝率
        data[f'wr{i}'] = float(tbody.select('td')[3].select_one('div').contents[0].strip())
        # モーター2連率
        data[f'mo{i}'] = float(tbody.select('td')[6].select_one('div').contents[0].strip())
        # F数
        f_text = tbody.select('td')[2].text.strip()
        data[f'f{i}'] = int(re.search(r'F(\d)', f_text).group(1)) if 'F' in f_text else 0
        # 平均ST
        st_text = tbody.select('td')[2].select_one('div').contents[-1].strip()
        data[f'st{i}'] = float(st_text) if st_text != '-' else 0.20

    # 2. 直前情報ページ (展示タイム, 風速)
    url_before = f"https://www.boatrace.jp/owpc/pc/race/beforeinfo?rno={rno}&jcd={jcd}&hd={date}"
    soup_before = get_soup(url_before)
    
    # 風速
    wind_elem = soup_before.select_one('.is-wind')
    data['wind'] = float(re.search(r'(\d+)m', wind_elem.text).group(1)) if wind_elem else 0.0
    
    # 展示タイム (ex)
    ex_table = soup_before.select_one('div.is-overflow table')
    if ex_table:
        rows = ex_table.select('tbody tr')
        for i in range(1, 7):
            ex_val = rows[i-1].select('td')[4].text.strip()
            data[f'ex{i}'] = float(ex_val) if ex_val != '-' else 6.80
    else:
        return None # 展示がまだの場合はスキップ
    
    return data

def engineer_features(df):
    """添付されたモデルと同じ特徴量エンジニアリング"""
    for i in range(1, 7):
        df[f'power_idx_{i}'] = df[f'wr{i}'] * (1.0 / (df[f'st{i}'] + 0.01))
    for i in range(1, 6):
        df[f'st_gap_{i}_{i+1}'] = df[f'st{i+1}'] - df[f'st{i}']
        df[f'wr_gap_{i}_{i+1}'] = df[f'wr{i}'] - df[f'wr{i+1}']
    avg_wr = df[[f'wr{i}' for i in range(1, 7)]].mean(axis=1)
    df['wr_1_vs_avg'] = df['wr1'] / (avg_wr + 0.001)
    df['jcd'] = df['jcd'].astype('category')
    return df

# ==========================================
# 🚀 メイン実行
# ==========================================

def main():
    if not os.path.exists('status.json'):
        with open('status.json', 'w') as f: json.dump({"notified": [], "results": [], "total_balance": 0}, f)
    
    with open('status.json', 'r') as f: status = json.load(f)
    today = datetime.datetime.now().strftime('%Y%m%d')
    bst = lgb.Booster(model_file=MODEL_FILE)
    
    jcds = fetch_active_races(today)
    
    for jcd in jcds:
        for rno in range(1, 13):
            race_id = f"{today}_{jcd}_{rno}"
            
            # --- 勝負判断 ---
            if not any(n['id'] == race_id for n in status["notified"]):
                try:
                    raw_data = scrape_race_data(jcd, rno, today)
                    if not raw_data: continue # 展示前
                    
                    df = pd.DataFrame([raw_data])
                    df = engineer_features(df)
                    
                    # 特徴量の並び順を学習時に合わせる(finalize_model.py参照)
                    features = ['jcd', 'rno', 'wind', 'wr_1_vs_avg']
                    for i in range(1, 7): features.extend([f'wr{i}', f'st{i}', f'ex{i}', f'power_idx_{i}'])
                    for i in range(1, 6): features.extend([f'st_gap_{i}_{i+1}', f'wr_gap_{i}_{i+1}'])
                    
                    probs = bst.predict(df[features])[0]
                    best_idx = np.argmax(probs)
                    prob = probs[best_idx]
                    combo = COMBOS[best_idx]
                    
                    # オッズ取得
                    res_odds = requests.get(f"https://www.boatrace.jp/owpc/pc/race/odds2t?rno={rno}&jcd={jcd}&hd={today}", headers=HEADERS)
                    soup_odds = BeautifulSoup(res_odds.text, 'html.parser')
                    # オッズ抽出(簡易)
                    odds = 1.0
                    for table in soup_odds.select('table.is-p_auto'):
                        for tr in table.select('tbody tr'):
                            if tr.select('td')[0].text.strip() == combo.split('-')[1] and tr.parent.parent.parent.select_one('thead').text.strip() == combo.split('-')[0]:
                                odds = float(tr.select('td')[1].text.strip())

                    ev = prob * odds
                    
                    if ev > 1.2 and prob > 0.4: # 条件
                        prompt = f"的中率{prob*100:.1f}%、期待値{ev:.2f}の「{combo}」は買いですか？"
                        res_gemini = model_gemini.generate_content(prompt).text
                        
                        if "買い" in res_gemini or "強気" in res_gemini:
                            live_url = f"https://www.boatrace.jp/owpc/pc/race/videolive?jcd={jcd}&hd={today}"
                            discord.post(content=f"🚀 **勝負！ {jcd}#{rno}R**\n買い目: {combo}\n{res_gemini}\n📺 {live_url}")
                            status["notified"].append({"id": race_id, "jcd": jcd, "rno": rno, "combo": combo, "amount": BET_AMOUNT})
                except Exception as e:
                    print(f"Error prediction {race_id}: {e}")

            # --- 結果確認 ---
            for task in status["notified"]:
                if any(r['id'] == task['id'] for r in status["results"]): continue
                
                try:
                    url_res = f"https://www.boatrace.jp/owpc/pc/race/raceresult?rno={task['rno']}&jcd={task['jcd']}&hd={today}"
                    soup_res = get_soup(url_res)
                    table = soup_res.select_one('table.is-w600') # 配当表
                    if table:
                        found_res = None; payout = 0
                        for tr in table.select('tr'):
                            if '2連単' in tr.text:
                                found_res = tr.select('td')[0].text.strip().replace(' ', '')
                                payout = int(tr.select('td')[1].text.strip().replace('¥', '').replace(',', ''))
                        
                        if found_res:
                            hit = (found_res == task['combo'])
                            profit = (payout * (task['amount'] // 100)) - task['amount'] if hit else -task['amount']
                            status["total_balance"] += profit
                            discord.post(content=f"🏁 **結果: {task['id']}**\n{found_res} ({'✅的中' if hit else '❌不的中'})\n収支: {profit:+}円 / 通算: {status['total_balance']:,}円")
                            status["results"].append({"id": task["id"]})
                except Exception as e:
                    print(f"Error result {task['id']}: {e}")

    with open('status.json', 'w') as f: json.dump(status, f, indent=4)

if __name__ == "__main__":
    main()