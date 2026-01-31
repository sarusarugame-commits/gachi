import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import zipfile
import time
import random
from itertools import permutations
import json

# ★ OpenAIライブラリ必須
OPENAI_AVAILABLE = False
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    pass

_GROQ_CLIENT = None

def get_groq_client():
    global _GROQ_CLIENT
    if not OPENAI_AVAILABLE: return None
    
    if _GROQ_CLIENT is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key: return None
        try:
            # Groqのエンドポイントを指定して初期化
            _GROQ_CLIENT = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=api_key,
                max_retries=3, 
                timeout=20.0
            )
        except: return None
    return _GROQ_CLIENT

MODEL_FILE = "boat_race_model_3t.txt"
AI_MODEL = None

# ★厳選設定: 閾値 4.0%
STRATEGY_DEFAULT = {'th': 0.040, 'k': 5}
STRATEGY = {}

def load_model():
    global AI_MODEL
    if AI_MODEL is None:
        if os.path.exists(MODEL_FILE):
            print(f"📂 モデルファイルを検出: {MODEL_FILE}")
            AI_MODEL = lgb.Booster(model_file=MODEL_FILE)
        elif os.path.exists(MODEL_FILE.replace(".txt", ".zip")):
            print(f"📦 ZIPモデルを解凍中: {MODEL_FILE.replace('.txt', '.zip')}")
            with zipfile.ZipFile(MODEL_FILE.replace(".txt", ".zip"), 'r') as z:
                z.extractall(".")
            AI_MODEL = lgb.Booster(model_file=MODEL_FILE)
        else:
            raise FileNotFoundError(f"モデルファイル '{MODEL_FILE}' が見つかりません。")
    return AI_MODEL

def generate_reason_with_groq(jcd, combo, prob, raw_data, odds):
    """
    OpenAI互換クライアントでGroq APIを叩く（オッズ分析付き）
    """
    client = get_groq_client()
    if not client:
        return f"AI推奨（自信度{prob}%）"

    models = ["meta-llama/llama-4-scout-17b-16e-instruct", "llama-3.3-70b-versatile"]
    selected_model = random.choice(models)

    players_info = ""
    for i in range(1, 7):
        s = str(i)
        wr = raw_data.get(f'wr{s}', 0.0)
        mo = raw_data.get(f'mo{s}', 0.0)
        ex = raw_data.get(f'ex{s}', 0.0)
        st = raw_data.get(f'st{s}', 0.0)
        players_info += f"{i}号艇: 勝率{wr:.2f} 機力{mo:.1f} 展示{ex:.2f} ST{st:.2f}\n"

    # オッズ分析テキスト
    odds_info = f"{odds}倍" if odds else "不明"
    expectation = "不明"
    if odds:
        ev = (float(prob) / 100) * odds
        expectation = f"{ev:.2f}"

    prompt = f"""
    あなたは辛口のボートレース投資家です。
    以下のデータと「現在のオッズ」を分析し、買い目「{combo}」が投資としてアリかナシか、40文字以内で断言してください。
    
    [レース環境]
    会場:{jcd}場 風:{raw_data.get('wind', 0)}m
    {players_info}
    
    [AI予測]
    推奨:{combo}
    的中率:{prob}%
    
    [オッズ分析]
    現在オッズ: {odds_info}
    期待値指数: {expectation} (目安1.0以上)
    """

    try:
        time.sleep(2.0)
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "あなたはプロの舟券師です。"},
                {"role": "user", "content": prompt}
            ],
            model=selected_model, 
            temperature=0.7,
            max_tokens=100,
        )
        return chat_completion.choices[0].message.content.strip()

    except Exception as e:
        print(f"❌ Groq API Error ({selected_model}): {e}")
        return f"AI解説取得エラー"

def attach_reason(results, raw, odds_value=None):
    if not results: return
    
    best_bet = results[0]
    combo = best_bet['combo']
    prob = best_bet['prob']
    jcd = raw.get('jcd', 0)
    
    reason_msg = generate_reason_with_groq(
        jcd, combo, prob, raw, odds_value
    )
    
    for rank, item in enumerate(results):
        if rank == 0:
            item['reason'] = reason_msg
            item['odds'] = odds_value
        else:
            item['reason'] = "同上（抑え）"

def predict_race(raw, odds_data=None):
    model = load_model()
    
    jcd = raw.get('jcd', 0)
    wind = raw.get('wind', 0.0)
    rno = raw.get('rno', 0)
    
    strat = STRATEGY.get(jcd, STRATEGY_DEFAULT)
    
    ex_values = [raw.get(f'ex{i}', 0) for i in range(1, 7)]
    if sum(ex_values) == 0:
        return []

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
    except Exception as e:
        print(f"❌ {jcd}場{rno}R: 予測エラー {e}")
        return []

    b = df_race['boat_no'].values
    combos = []
    for i, j, k in permutations(range(6), 3):
        score = p1[i] * p2[j] * p3[k]
        combos.append({
            'combo': f"{b[i]}-{b[j]}-{b[k]}",
            'score': score
        })
    combos.sort(key=lambda x: x['score'], reverse=True)
    
    best_bet = combos[0]

    if best_bet['score'] < strat['th']:
        if best_bet['score'] > 0.035:
             print(f"📉 {jcd}場{rno}R: スコア不足 (Best: {best_bet['score']*100:.2f}%) -> {best_bet['combo']}")
        return []

    results = []
    for rank, item in enumerate(combos[:strat['k']]):
        prob_percent = item['score'] * 100
        results.append({
            'combo': item['combo'],
            'type': f"ランク{rank+1}",
            'profit': "計算中",
            'prob': f"{prob_percent:.1f}",
            'roi': 0,
            'reason': "待機中...",
            'deadline': raw.get('deadline_time', '不明')
        })
    return results
