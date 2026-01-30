import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import zipfile
from itertools import permutations

MODEL_FILE = "boat_race_model_3t.txt"
AI_MODEL = None

STRATEGY = {
    1:  {'th': 0.065, 'k': 1}, 2:  {'th': 0.050, 'k': 5}, 3:  {'th': 0.060, 'k': 8},
    4:  {'th': 0.050, 'k': 5}, 5:  {'th': 0.040, 'k': 1}, 7:  {'th': 0.065, 'k': 1},
    8:  {'th': 0.070, 'k': 5}, 9:  {'th': 0.055, 'k': 1}, 10: {'th': 0.060, 'k': 8},
    11: {'th': 0.045, 'k': 1}, 12: {'th': 0.060, 'k': 1}, 13: {'th': 0.040, 'k': 1},
    15: {'th': 0.065, 'k': 1}, 16: {'th': 0.055, 'k': 1}, 18: {'th': 0.070, 'k': 1},
    19: {'th': 0.065, 'k': 1}, 20: {'th': 0.070, 'k': 8}, 21: {'th': 0.060, 'k': 1},
    22: {'th': 0.055, 'k': 1},
}

def load_model():
    global AI_MODEL
    if AI_MODEL is None:
        if os.path.exists(MODEL_FILE):
            AI_MODEL = lgb.Booster(model_file=MODEL_FILE)
        elif os.path.exists(MODEL_FILE.replace(".txt", ".zip")):
            with zipfile.ZipFile(MODEL_FILE.replace(".txt", ".zip"), 'r') as z:
                z.extractall(".")
            AI_MODEL = lgb.Booster(model_file=MODEL_FILE)
    return AI_MODEL

def predict_race(raw, odds_data=None):
    model = load_model()
    if model is None: return []

    jcd = raw.get('jcd', 0)
    wind = raw.get('wind', 0.0)
    if jcd not in STRATEGY: return []
    
    # 展示タイムなしはスキップ
    if sum([raw.get(f'ex{i}', 0) for i in range(1, 7)]) == 0: return []

    rows = []
    for i in range(1, 7):
        s = str(i)
        rows.append({
            'jcd': jcd, 'wind': wind, 'boat_no': i,
            'pid': raw.get(f'pid{s}', 0), 'wr': raw.get(f'wr{s}', 0.0),
            'mo': raw.get(f'mo{s}', 0.0), 'ex': raw.get(f'ex{s}', 0.0),
            'st': raw.get(f'st{s}', 0.20), 'f': raw.get(f'f{s}', 0),
        })
    df_race = pd.DataFrame(rows)

    for col in ['wr', 'mo', 'ex', 'st']:
        mean = df_race[col].mean()
        std = df_race[col].std()
        df_race[f'{col}_z'] = (df_race[col] - mean) / (std + 1e-6)

    df_race['jcd'] = df_race['jcd'].astype('category')
    df_race['pid'] = df_race['pid'].astype('category')
    
    features = [
        'jcd', 'boat_no', 'wind', 'pid',
        'wr', 'mo', 'ex', 'st', 'f',
        'wr_z', 'mo_z', 'ex_z', 'st_z'
    ]

    try:
        preds = model.predict(df_race[features])
        if preds.shape[1] < 3: return []
        p1, p2, p3 = preds[:, 0], preds[:, 1], preds[:, 2]
    except: return []

    b = df_race['boat_no'].values
    combos = []
    for i, j, k in permutations(range(6), 3):
        score = p1[i] * p2[j] * p3[k]
        combos.append({
            'combo': f"{b[i]}-{b[j]}-{b[k]}",
            'score': score
        })
    combos.sort(key=lambda x: x['score'], reverse=True)
    
    strat = STRATEGY[jcd]
    best_bet = combos[0]

    if best_bet['score'] >= strat['th']:
        results = []
        for rank, item in enumerate(combos[:strat['k']]):
            # 自信度をパーセント表示に
            prob_percent = item['score'] * 100
            
            # AI解説の生成
            comment = "AI推奨"
            if prob_percent > 10:
                comment = "🔥 超鉄板級！的中率極大"
            elif prob_percent > 5:
                comment = "✨ かなり有望！本命サイド"
            elif prob_percent > 2:
                comment = "👍 妙味あり！狙い目"
            
            # 理由付け
            reason_msg = f"{jcd}場の合格ライン({strat['th']})をクリア。AI評価「{comment}」"

            results.append({
                'combo': item['combo'],
                'type': f"ランク{rank+1}",
                'profit': "計算中", # オッズ未取得のため
                'prob': f"{prob_percent:.1f}", # 小数点1位まで表示
                'roi': 0,
                'reason': reason_msg
            })
        return results

    return []
