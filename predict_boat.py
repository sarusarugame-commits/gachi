import pandas as pd
import numpy as np
import lightgbm as lgb
import requests
from bs4 import BeautifulSoup
import datetime
import os
import re
from itertools import permutations
import time

# ==========================================
# ⚙️ 設定エリア
# ==========================================
MODEL_PATH = "boat_race_model_3t.txt"

# 予想したい日（Noneなら「今日」）
TARGET_DATE = None  # 例: "20260130"

# 【会場別】最適戦略ポートフォリオ
# format: JCD: {'th': 閾値, 'k': 購入点数}
# シミュレーション結果に基づき設定
STRATEGY = {
    1:  {'th': 0.065, 'k': 1},  # 桐生
    2:  {'th': 0.050, 'k': 5},  # 戸田
    3:  {'th': 0.060, 'k': 8},  # 江戸川
    4:  {'th': 0.050, 'k': 5},  # 平和島
    5:  {'th': 0.040, 'k': 1},  # 多摩川
    7:  {'th': 0.065, 'k': 1},  # 蒲郡
    8:  {'th': 0.070, 'k': 5},  # 常滑
    9:  {'th': 0.055, 'k': 1},  # 津
    10: {'th': 0.060, 'k': 8},  # 三国 (稼ぎ頭)
    11: {'th': 0.045, 'k': 1},  # びわこ
    12: {'th': 0.060, 'k': 1},  # 住之江
    13: {'th': 0.040, 'k': 1},  # 尼崎
    15: {'th': 0.065, 'k': 1},  # 丸亀
    16: {'th': 0.055, 'k': 1},  # 児島
    18: {'th': 0.070, 'k': 1},  # 徳山
    19: {'th': 0.065, 'k': 1},  # 下関
    20: {'th': 0.070, 'k': 8},  # 若松
    21: {'th': 0.060, 'k': 1},  # 芦屋
    22: {'th': 0.055, 'k': 1},  # 福岡
}

# ==========================================
# 1. スクレイピング関数
# ==========================================
def get_soup(url):
    try:
        res = requests.get(url, timeout=5)
        res.encoding = res.apparent_encoding
        return BeautifulSoup(res.text, 'html.parser')
    except: return None

def clean_text(text):
    return text.replace("\n", "").replace(" ", "").strip()

def scrape_race_info(jcd, rno, date_str):
    base_url = "https://www.boatrace.jp/owpc/pc/race"
    url_lst = f"{base_url}/racelist?rno={rno}&jcd={jcd:02d}&hd={date_str}"
    url_bef = f"{base_url}/beforeinfo?rno={rno}&jcd={jcd:02d}&hd={date_str}"
    
    soup_lst = get_soup(url_lst)
    soup_bef = get_soup(url_bef)
    
    if not soup_lst: return None
    
    rows = []
    wind = 0.0
    if soup_bef:
        try:
            w_txt = soup_bef.select_one(".weather1_bodyUnitLabelData").text
            m = re.search(r"(\d+)", clean_text(w_txt))
            if m: wind = float(m.group(1))
        except: pass

    for i in range(1, 7):
        row = {
            'race_id': f"{date_str}_{jcd:02d}_{rno:02d}",
            'date': int(date_str),
            'jcd': jcd,
            'wind': wind,
            'boat_no': i,
            'pid': 0, 'wr': 0.0, 'mo': 0.0, 'ex': 0.0, 'st': 0.20, 'f': 0
        }
        try:
            tbody = soup_lst.select("tbody.is-fs12")[i-1]
            pid_m = re.search(r"(\d{4})", tbody.select_one(".is-fs11").text)
            if pid_m: row['pid'] = int(pid_m.group(1))
            tds = tbody.select("td")
            if len(tds) > 4:
                m = re.search(r"(\d\.\d{2})", clean_text(tds[4].text))
                if m: row['wr'] = float(m.group(1))
            if len(tds) > 6:
                txt = clean_text(tds[6].text)
                m = re.search(r"(0\.\d{2})", txt)
                if m: row['st'] = float(m.group(1))
                mf = re.search(r"F(\d+)", txt)
                if mf: row['f'] = int(mf.group(1))
            if len(tds) > 7:
                m = re.search(r"(\d{2}\.\d{2})", clean_text(tds[7].text))
                if m: row['mo'] = float(m.group(1))
        except: pass

        if soup_bef:
            try:
                boat_td = soup_bef.select_one(f"td.is-boatColor{i}")
                if boat_td:
                    tr = boat_td.find_parent("tr")
                    tds = tr.select("td")
                    for td in tds[4:]:
                        val = clean_text(td.text)
                        if re.match(r"^\d\.\d{2}$", val):
                            fval = float(val)
                            if 6.0 <= fval <= 7.5:
                                row['ex'] = fval
                                break
            except: pass
        rows.append(row)
    return pd.DataFrame(rows)

# ==========================================
# 2. 予測関数
# ==========================================
def predict_race(model, df_race):
    for col in ['wr', 'mo', 'ex', 'st']:
        mean = df_race[col].mean()
        std = df_race[col].std()
        if std == 0: std = 1e-6
        df_race[f'{col}_z'] = (df_race[col] - mean) / std

    df_race['jcd'] = df_race['jcd'].astype('category')
    df_race['pid'] = df_race['pid'].astype('category')
    
    features = [
        'jcd', 'boat_no', 'wind', 'pid',
        'wr', 'mo', 'ex', 'st', 'f',
        'wr_z', 'mo_z', 'ex_z', 'st_z'
    ]
    
    preds = model.predict(df_race[features])
    df_race['p1'] = preds[:, 0]
    df_race['p2'] = preds[:, 1]
    df_race['p3'] = preds[:, 2]
    
    p1 = df_race['p1'].values
    p2 = df_race['p2'].values
    p3 = df_race['p3'].values
    b = df_race['boat_no'].values
    
    combos = []
    for i, j, k in permutations(range(6), 3):
        score = p1[i] * p2[j] * p3[k]
        combos.append({
            'combo': f"{b[i]}-{b[j]}-{b[k]}",
            'score': score
        })
    combos.sort(key=lambda x: x['score'], reverse=True)
    return combos

# ==========================================
# メイン実行
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"❌ モデルファイルが見つかりません: {MODEL_PATH}")
        exit()

    print("📂 モデルを読み込んでいます...")
    model = lgb.Booster(model_file=MODEL_PATH)
    
    if TARGET_DATE is None:
        today = datetime.date.today()
        date_str = today.strftime("%Y%m%d")
    else:
        date_str = TARGET_DATE
        
    print(f"🚀 {date_str} 本日の勝負レースを探索します...")
    print("-" * 65)

    hit_count = 0
    total_cost = 0

    # 全場チェック
    # 戦略リストにある場だけチェックしてもいいが、一応全場見る
    for jcd in range(1, 25):
        # 戦略が定義されていない場はスキップ（利益が出ない場）
        if jcd not in STRATEGY:
            continue
            
        strat = STRATEGY[jcd]
        
        for rno in range(1, 13):
            # サーバー負荷軽減
            time.sleep(0.05)
            
            # データ取得
            df = scrape_race_info(jcd, rno, date_str)
            if df is None or len(df) == 0: continue
            
            # 直前情報なし(ex=0)はスキップ
            if df['ex'].sum() == 0: continue

            try:
                top_combos = predict_race(model, df)
                best_score = top_combos[0]['score']
                
                # 戦略の閾値を超えているか？
                if best_score >= strat['th']:
                    hit_count += 1
                    cost = strat['k'] * 100
                    total_cost += cost
                    
                    print(f"🔥 {jcd:02}場 {rno:02}R | 自信度:{best_score:.4f} (基準 {strat['th']}) | {strat['k']}点買い")
                    print(f"   [本命] {top_combos[0]['combo']}")
                    
                    if strat['k'] > 1:
                        print(f"   [紐  ] {', '.join([c['combo'] for c in top_combos[1:strat['k']] ])}")
                    
                    print("-" * 65)
                    
            except: pass

    if hit_count == 0:
        print("🍵 現在、条件を満たすレースはありません。直前情報の更新を待ってください。")
    else:
        print(f"💰 合計 {hit_count} レース推奨 | 推定投資額: {total_cost:,} 円")
        print("   Good Luck!")
