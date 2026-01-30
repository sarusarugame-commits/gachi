import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import zipfile
from itertools import permutations
import json

# ★ GROQクライアントの準備
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

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

def generate_reason_with_groq(jcd, boat_no_list, combo, prob, raw_data):
    """
    Groq API (Llama 4 Scout) を使って解説を生成
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not GROQ_AVAILABLE or not api_key:
        return f"基準クリア（自信度{prob}%）"

    try:
        client = Groq(api_key=api_key)
        
        # 選手データの要約を作成
        players_info = ""
        for i in range(1, 7):
            s = str(i)
            pid = raw_data.get(f'pid{s}', 0)
            wr = raw_data.get(f'wr{s}', 0.0)
            mo = raw_data.get(f'mo{s}', 0.0)
            ex = raw_data.get(f'ex{s}', 0.0)
            st = raw_data.get(f'st{s}', 0.0)
            players_info += f"{i}号艇: 勝率{wr:.2f} モータ{mo:.1f} 展示{ex:.2f} ST{st:.2f}\n"

        prompt = f"""
        あなたは「Llama 4 Scout」です。鋭い観察眼を持つボートレースのスカウトマンとして振る舞ってください。
        以下のレースデータに基づき、なぜ買い目「{combo}」が激アツなのか、50文字以内でズバリ解説してください。
        データ（勝率、機力、展示）に基づいたプロの視点を入れてください。

        [レースデータ]
        会場: {jcd}場
        風速: {raw_data.get('wind', 0)}m
        {players_info}

        [AI予測]
        推奨買い目: {combo}
        当選確率: {prob}%
        """

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "あなたはLlama 4 Scoutです。的確で冷徹なボートレース分析官です。"},
                {"role": "user", "content": prompt}
            ],
            # ★ここを変更：Llama 4 Scoutを指定
            model="llama-4-scout-17b-16e-instruct", 
            temperature=0.7,
            max_tokens=100,
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        # 万が一モデルIDが微妙に異なる場合のエラーハンドリング
        print(f"Groq Error: {e}")
        return f"基準クリア（自信度{prob}%）"

def predict_race(raw, odds_data=None):
    model = load_model()
    if model is None: return []

    jcd = raw.get('jcd', 0)
    wind = raw.get('wind', 0.0)
    if jcd not in STRATEGY: return []
    
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
        # 解説生成
        reason_msg = generate_reason_with_groq(
            jcd, [int(x) for x in best_bet['combo'].split('-')], 
            best_bet['combo'], 
            f"{best_bet['score']*100:.1f}", 
            raw
        )
        
        for rank, item in enumerate(combos[:strat['k']]):
            prob_percent = item['score'] * 100
            current_reason = reason_msg if rank == 0 else "同上（抑え）"

            results.append({
                'combo': item['combo'],
                'type': f"ランク{rank+1}",
                'profit': "計算中",
                'prob': f"{prob_percent:.1f}",
                'roi': 0,
                'reason': current_reason,
                'deadline': raw.get('deadline_time', '不明')
            })
        return results

    return []
