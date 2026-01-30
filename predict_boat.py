import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import zipfile
import time
import random
from itertools import permutations
import json

# ★ GROQクライアントの準備
GROQ_AVAILABLE = False
try:
    from groq import Groq, APIConnectionError, RateLimitError
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# グローバル変数としてクライアントを保持
_GROQ_CLIENT = None

def get_groq_client():
    global _GROQ_CLIENT
    if not GROQ_AVAILABLE:
        return None
    
    if _GROQ_CLIENT is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if api_key:
            try:
                # タイムアウトを少し長めに設定
                _GROQ_CLIENT = Groq(
                    api_key=api_key, 
                    max_retries=2,
                    timeout=20.0 
                )
            except Exception as e:
                print(f"Groq Init Error: {e}")
                return None
    return _GROQ_CLIENT

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
    Groq API を使って解説を生成（モデルランダム切り替え版）
    """
    client = get_groq_client()
    if not client:
        return f"基準クリア（自信度{prob}%）"

    # ★ 2つのモデルを定義 (70BはLlama 3.3を使用)
    models = [
        "llama-4-scout-17b-16e-instruct", # ユーザー指定
        "llama-3.3-70b-versatile"         # 70Bモデル
    ]
    # ランダムに選択して負荷分散
    selected_model = random.choice(models)

    # 選手データの要約
    players_info = ""
    for i in range(1, 7):
        s = str(i)
        wr = raw_data.get(f'wr{s}', 0.0)
        mo = raw_data.get(f'mo{s}', 0.0)
        ex = raw_data.get(f'ex{s}', 0.0)
        st = raw_data.get(f'st{s}', 0.0)
        players_info += f"{i}号艇: 勝率{wr:.2f} 機力{mo:.1f} 展示{ex:.2f} ST{st:.2f}\n"

    prompt = f"""
    あなたはボートレースのプロ予想家です。
    以下のデータに基づき、買い目「{combo}」を推奨する理由を50文字以内で簡潔に述べよ。
    使用モデル: {selected_model}
    
    [データ]
    会場: {jcd}場, 風速: {raw_data.get('wind', 0)}m
    {players_info}
    [予測]
    推奨: {combo}, 確率: {prob}%
    """

    # リトライ処理（最大3回）
    for attempt in range(3):
        try:
            # ★ API制限対策: リクエスト前に少し待機 (1〜3秒)
            time.sleep(random.uniform(1.0, 3.0))

            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "あなたは的確なボートレース分析官です。"},
                    {"role": "user", "content": prompt}
                ],
                model=selected_model, 
                temperature=0.7,
                max_tokens=120,
            )
            return chat_completion.choices[0].message.content.strip()

        except Exception as e:
            # エラー内容を少し詳細に出力（デバッグ用）
            print(f"Groq Retry({attempt+1}/3) {selected_model}: {e}")
            if attempt < 2:
                time.sleep(5)  # エラー時は長めに待機 (5秒)
            else:
                return f"AI解説取得エラー（確率{prob}%）"

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
