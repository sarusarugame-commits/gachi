import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import zipfile
import time
import random
from itertools import permutations
import json

# ★ GROQクライアントの準備（エラーログ強化版）
GROQ_AVAILABLE = False
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("⚠️ 'groq' ライブラリが見つかりません。pip install groq を実行してください。")

_GROQ_CLIENT = None

def get_groq_client():
    global _GROQ_CLIENT
    if not GROQ_AVAILABLE:
        return None
    
    if _GROQ_CLIENT is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("⚠️ 環境変数 'GROQ_API_KEY' が設定されていません。")
            return None
            
        try:
            _GROQ_CLIENT = Groq(api_key=api_key, max_retries=0, timeout=10.0)
        except Exception as e:
            print(f"❌ Groqクライアント初期化エラー: {e}")
            return None
    return _GROQ_CLIENT

MODEL_FILE = "boat_race_model_3t.txt"
AI_MODEL = None

# ★【厳選設定】1日2〜3レースを狙うため、閾値を 0.040 (4.0%) に設定
# 点数(k)は5点のまま維持（的中率確保のため）
STRATEGY_DEFAULT = {'th': 0.040, 'k': 5}
STRATEGY = {
    # 特定の場だけ調整したい場合はここに記述
    # 例: 1: {'th': 0.045, 'k': 5}, 
}

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
            cwd_files = os.listdir(".")
            raise FileNotFoundError(f"モデルファイル '{MODEL_FILE}' が見つかりません。")
            
    return AI_MODEL

def generate_reason_with_groq(jcd, boat_no_list, combo, prob, raw_data):
    """
    Groq API を使って解説を生成
    """
    client = get_groq_client()
    if not client:
        return f"基準クリア（自信度{prob}%）"

    models = ["llama-4-scout-17b-16e-instruct", "llama-3.3-70b-versatile"]
    selected_model = random.choice(models)

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
    
    [データ]
    会場: {jcd}場, 風速: {raw_data.get('wind', 0)}m
    {players_info}
    [予測]
    推奨: {combo}, 確率: {prob}%
    """

    try:
        time.sleep(random.uniform(0.5, 1.5))
        
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "あなたは的確なボートレース分析官です。"},
                {"role": "user", "content": prompt}
            ],
            model=selected_model, 
            temperature=0.7,
            max_tokens=100,
        )
        return chat_completion.choices[0].message.content.strip()

    except Exception as e:
        print(f"⚠️ Groq API呼び出しエラー ({selected_model}): {e}")
        return f"AI推奨（自信度{prob}%）※解説生成失敗"

def attach_reason(results, raw):
    if not results: return
    
    best_bet = results[0]
    combo = best_bet['combo']
    prob = best_bet['prob']
    jcd = raw.get('jcd', 0)
    
    reason_msg = generate_reason_with_groq(
        jcd, [int(x) for x in combo.split('-')], 
        combo, prob, raw
    )
    
    for rank, item in enumerate(results):
        if rank == 0:
            item['reason'] = reason_msg
        else:
            item['reason'] = "同上（抑え）"

def predict_race(raw, odds_data=None):
    model = load_model()
    
    jcd = raw.get('jcd', 0)
    wind = raw.get('wind', 0.0)
    rno = raw.get('rno', 0)
    
    # デフォルト設定を使用
    strat = STRATEGY.get(jcd, STRATEGY_DEFAULT)
    
    ex_values = [raw.get(f'ex{i}', 0) for i in range(1, 7)]
    if sum(ex_values) == 0:
        print(f"⚠️ {jcd}場{rno}R: 展示タイムなし -> スキップ (Ex: {ex_values})")
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

    # 閾値チェック (ログ出力付き)
    if best_bet['score'] < strat['th']:
        # 3.5%以上なら「惜しい」ログを出す
        if best_bet['score'] > 0.035:
             print(f"📉 {jcd}場{rno}R: スコア不足 (Best: {best_bet['score']*100:.2f}% / 必要: {strat['th']*100:.1f}%) -> {best_bet['combo']}")
        return []

    # 合格
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
