import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import zipfile
import time
import random
from itertools import permutations
import json

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

# ★閾値設定を 0.050 (5%) に変更
# これにより、AIが「5%以上の確率で来る」と断言したもの以外はすべて切り捨てる
STRATEGY_DEFAULT = {'th': 0.050, 'k': 5}
STRATEGY = {}

def load_model():
    global AI_MODEL
    if AI_MODEL is None:
        if os.path.exists(MODEL_FILE):
            print(f"📂 モデルファイルを検出: {MODEL_FILE}")
            AI_MODEL = lgb.Booster(model_file=MODEL_FILE)
        elif os.path.exists(MODEL_FILE.replace(".txt", ".zip")):
            with zipfile.ZipFile(MODEL_FILE.replace(".txt", ".zip"), 'r') as z:
                z.extractall(".")
            AI_MODEL = lgb.Booster(model_file=MODEL_FILE)
        else:
            raise FileNotFoundError(f"モデルファイル '{MODEL_FILE}' が見つかりません。")
    return AI_MODEL

def generate_batch_reasons(jcd, bets_info, raw_data):
    """
    複数の買い目をまとめてAIに分析させ、個別のコメントを取得する
    """
    client = get_groq_client()
    if not client: return {}

    models = ["meta-llama/llama-4-scout-17b-16e-instruct", "llama-3.3-70b-versatile"]
    selected_model = random.choice(models)

    players_info = ""
    for i in range(1, 7):
        s = str(i)
        wr = raw_data.get(f'wr{s}', 0.0)
        mo = raw_data.get(f'mo{s}', 0.0)
        players_info += f"{i}号艇:勝率{wr:.2f}/機力{mo:.1f} "
    
    bets_text = ""
    for b in bets_info:
        odds_str = f"{b['odds']}倍" if b['odds'] else "不明"
        ev_str = f"{b['ev']:.2f}" if b['ev'] else "-"
        bets_text += f"- {b['combo']}: 確率{b['prob']}% オッズ{odds_str} (期待値{ev_str})\n"

    prompt = f"""
    あなたは辛口のボートレース投資家です。
    以下の{jcd}場のレースの「買い目リスト」を評価してください。
    
    [選手データ]
    {players_info}
    
    [買い目リスト]
    {bets_text}
    
    【重要】
    各買い目に対して、オッズと確率のバランス（期待値）を見た上で、「投資すべきか」「危険か」「妙味ありか」など、
    一言ずつ（20文字以内）で鋭いコメントを付けてください。
    
    出力形式は買い目とコメントをコロン区切りで1行ずつ。
    例:
    1-2-3: 本命だが配当安すぎ、見送り推奨。
    1-2-4: このオッズなら狙う価値あり。
    """

    try:
        time.sleep(2.0)
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "あなたは実利重視のプロ舟券師です。"},
                {"role": "user", "content": prompt}
            ],
            model=selected_model, 
            temperature=0.7,
            max_tokens=300,
        )
        response_text = chat_completion.choices[0].message.content.strip()
        
        comments = {}
        for line in response_text.split('\n'):
            if ':' in line:
                parts = line.split(':', 1)
                combo_raw = parts[0].strip()
                comment = parts[1].strip()
                comments[combo_raw] = comment
        return comments
    except Exception as e:
        print(f"❌ Groq API Error: {e}")
        return {}

def attach_reason(results, raw, odds_map=None):
    if not results: return
    if odds_map is None: odds_map = {}
    
    jcd = raw.get('jcd', 0)
    
    bets_to_analyze = []
    for item in results:
        combo = item['combo']
        prob = float(item['prob'])
        odds = odds_map.get(combo)
        
        ev = None
        if odds:
            ev = (prob / 100) * odds
            item['odds'] = odds
            item['ev'] = ev
        
        bets_to_analyze.append({'combo': combo, 'prob': prob, 'odds': odds, 'ev': ev})

    ai_comments = generate_batch_reasons(jcd, bets_to_analyze, raw)
    
    for item in results:
        combo = item['combo']
        ev_val = item.get('ev')
        ai_comment = ai_comments.get(combo)
        ev_str = f"(EV:{ev_val:.2f})" if ev_val else ""
        
        if ai_comment:
            item['reason'] = f"{ai_comment} {ev_str}"
        else:
            if ev_val:
                if ev_val >= 1.5: item['reason'] = f"🔥超抜期待値！ {ev_str}"
                elif ev_val >= 1.0: item['reason'] = f"配当妙味あり。 {ev_str}"
                elif ev_val >= 0.8: item['reason'] = f"抑え妥当。 {ev_str}"
                else: item['reason'] = f"オッズ辛い。 {ev_str}"
            else:
                item['reason'] = "オッズ不明"

def predict_race(raw, odds_data=None):
    model = load_model()
    jcd = raw.get('jcd', 0)
    wind = raw.get('wind', 0.0)
    rno = raw.get('rno', 0)
    
    # 厳選設定: 閾値 5.0%
    strat = STRATEGY.get(jcd, STRATEGY_DEFAULT)
    
    ex_values = [raw.get(f'ex{i}', 0) for i in range(1, 7)]
    if sum(ex_values) == 0: return []

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
    
    features = ['jcd', 'boat_no', 'wind', 'pid', 'wr', 'mo', 'ex', 'st', 'f', 'wr_z', 'mo_z', 'ex_z', 'st_z']

    try:
        preds = model.predict(df_race[features])
        p1, p2, p3 = preds[:, 0], preds[:, 1], preds[:, 2]
    except: return []

    b = df_race['boat_no'].values
    combos = []
    for i, j, k in permutations(range(6), 3):
        score = p1[i] * p2[j] * p3[k]
        combos.append({'combo': f"{b[i]}-{b[j]}-{b[k]}", 'score': score})
    combos.sort(key=lambda x: x['score'], reverse=True)
    
    best_bet = combos[0]

    # ★ 1番自信のある買い目ですら 5.0% (0.05) 未満なら、レースごと見送り
    # これで「自信のないレース」には一切手を出さなくなります
    if best_bet['score'] < 0.05:
        # print(f"📉 スコア不足: {best_bet['score']*100:.1f}% < 5%")
        return []

    results = []
    for rank, item in enumerate(combos[:strat['k']]):
        # ★ 個別の買い目も 5.0% 未満ならカット
        if item['score'] < 0.05:
            continue
            
        results.append({
            'combo': item['combo'],
            'type': f"ランク{rank+1}",
            'profit': "計算中",
            'prob': f"{item['score']*100:.1f}",
            'roi': 0,
            'reason': "待機中...",
            'deadline': raw.get('deadline_time', '不明')
        })
        
    return results
