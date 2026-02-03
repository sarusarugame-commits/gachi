import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import zipfile
import time
import random
from itertools import permutations

# ==========================================
# ⚙️ 設定: 回収率130%ロジック
# ==========================================
MODEL_FILE = "boatrace_model.txt"

# フィルタ設定
MIN_PROB_THRESHOLD = 0.03       # 確率3%以上のみ候補にする
MAX_BETS_PER_RACE = 6           # 1レース最大6点
CALC_ODDS_CAP = 40.0            # オッズキャップ

# 会場ごとの最適EV閾値 (2023-2025分析結果)
BEST_EV_THRESHOLDS = {
    1: 1.4,  2: 1.8,  3: 99.9, 4: 1.2,  5: 99.9, 6: 1.5,
    7: 1.8,  8: 99.9, 9: 1.3,  10: 1.8, 11: 2.0, 12: 99.9,
    13: 2.0, 14: 1.4, 15: 1.8, 16: 1.8, 17: 2.0, 18: 2.0,
    19: 1.6, 20: 1.8, 21: 1.4, 22: 1.4, 23: 1.3, 24: 99.9
}

# ==========================================
# 🤖 Groq / OpenAI 設定
# ==========================================
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
                max_retries=3, timeout=20.0
            )
        except: return None
    return _GROQ_CLIENT

AI_MODEL = None
def load_model():
    global AI_MODEL
    if AI_MODEL is None:
        # 新モデル優先
        if os.path.exists(MODEL_FILE):
            print(f"📂 モデルファイルを検出: {MODEL_FILE}")
            AI_MODEL = lgb.Booster(model_file=MODEL_FILE)
        elif os.path.exists("boat_race_model_3t.txt"):
            print(f"📂 モデルファイルを検出(旧): boat_race_model_3t.txt")
            AI_MODEL = lgb.Booster(model_file="boat_race_model_3t.txt")
        else:
            raise FileNotFoundError("モデルファイルが見つかりません")
    return AI_MODEL

def to_float(val):
    try:
        if val is None or val == "": return 0.0
        return float(val)
    except: return 0.0

# ==========================================
# 🔮 1. 候補出し (確率計算)
# ==========================================
def predict_race(raw):
    """
    確率3%以上の買い目を広めに抽出して返す。
    オッズによる絞り込みはまだ行わない。
    """
    model = load_model()
    jcd = raw.get('jcd', 0)
    wind = to_float(raw.get('wind', 0.0))

    rows = []
    ex_list = []
    for i in range(1, 7):
        s = str(i)
        val_ex = to_float(raw.get(f'ex{s}', 0))
        ex_list.append(val_ex)
        rows.append({
            'jcd': jcd, 'wind': wind, 'boat_no': i,
            'pid': raw.get(f'pid{s}', 0), 
            'wr': to_float(raw.get(f'wr{s}', 0)),
            'mo': to_float(raw.get(f'mo{s}', 0)), 
            'ex': val_ex,
            'st': to_float(raw.get(f'st{s}', 0.20)), 
            'f': to_float(raw.get(f'f{s}', 0)),
        })
    
    if sum(ex_list) == 0: return []

    df_race = pd.DataFrame(rows)
    for col in ['wr', 'mo', 'ex', 'st']:
        mean_val = df_race[col].mean()
        std_val = df_race[col].std()
        if std_val == 0: std_val = 1e-6
        df_race[f'{col}_z'] = (df_race[col] - mean_val) / std_val

    df_race['jcd'] = df_race['jcd'].astype('category')
    df_race['pid'] = df_race['pid'].astype('category')
    
    features = ['jcd', 'boat_no', 'pid', 'wind', 'wr', 'mo', 'ex', 'st', 'f', 'wr_z', 'mo_z', 'ex_z', 'st_z']
    
    try:
        preds = model.predict(df_race[features])
        p1, p2, p3 = preds[:, 0], preds[:, 1], preds[:, 2]
    except: return []

    # 自信度チェック (自信がないレースはここで弾く)
    if max(p1) < 0.20: return []

    b = df_race['boat_no'].values
    candidates = []
    for i, j, k in permutations(range(6), 3):
        score = p1[i] * p2[j] * p3[k]
        if score >= MIN_PROB_THRESHOLD:
            candidates.append({
                'combo': f"{b[i]}-{b[j]}-{b[k]}",
                'raw_prob': score,
                'prob': round(score * 100, 1)
            })
    
    candidates.sort(key=lambda x: x['raw_prob'], reverse=True)
    return candidates[:20] # EV計算用に上位20件を返す

# ==========================================
# 💰 2. EVフィルタ (ここで厳選)
# ==========================================
def filter_and_sort_bets(candidates, odds_map, jcd):
    """
    候補リストにオッズを当てて、EV閾値を超えたものだけを返す
    """
    threshold = BEST_EV_THRESHOLDS.get(jcd, 99.9)
    if threshold >= 99.0: return [] # 見送り会場

    final_bets = []
    for bet in candidates:
        combo = bet['combo']
        prob = bet['raw_prob']
        
        real_odds = odds_map.get(combo, 0.0)
        if real_odds == 0: continue
        
        calc_odds = min(real_odds, CALC_ODDS_CAP)
        ev = prob * calc_odds
        
        if ev >= threshold:
            bet['odds'] = real_odds
            bet['ev'] = ev
            # reasonは後でGroqで上書きされるが、基本形を入れておく
            bet['reason'] = f"EV:{ev:.2f} (基準{threshold})"
            final_bets.append(bet)
            
    final_bets.sort(key=lambda x: x['ev'], reverse=True)
    return final_bets[:MAX_BETS_PER_RACE]

# ==========================================
# 📝 3. 解説生成
# ==========================================
def generate_batch_reasons(jcd, bets_info, raw_data):
    client = get_groq_client()
    if not client: return {}
    
    # 選手情報
    players_info = ""
    for i in range(1, 7):
        players_info += f"{i}号艇:勝率{raw_data.get(f'wr{i}',0)} "

    bets_text = ""
    for b in bets_info:
        bets_text += f"- {b['combo']}: 確率{b['prob']}% オッズ{b['odds']} (EV{b['ev']:.2f})\n"

    prompt = f"""
    ボートレースのベテラン予想家として、以下の{jcd}場の推奨買い目を解説してください。
    
    [選手] {players_info}
    [買い目] {bets_text}
    
    【指示】
    各買い目について、なぜチャンスなのか **30文字以内** で平易にコメントしてください。
    必ず **【勝負】** か **【見送り】** で始めてください。
    
    出力形式:
    1-2-3: 【勝負】 1番が盤石！配当も美味しく狙い目。
    """
    
    try:
        chat = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile", temperature=0.7, max_tokens=400
        )
        text = chat.choices[0].message.content
        comments = {}
        for line in text.split('\n'):
            if ':' in line:
                p = line.split(':', 1)
                comments[p[0].strip()] = p[1].strip()
        return comments
    except: return {}

def attach_reason(results, raw, odds_map):
    if not results: return
    jcd = raw.get('jcd', 0)
    
    ai_comments = generate_batch_reasons(jcd, results, raw)
    
    for item in results:
        combo = item['combo']
        ai_msg = ai_comments.get(combo)
        if ai_msg:
            item['reason'] = f"{ai_msg} (EV:{item['ev']:.2f})"
        else:
            item['reason'] = f"【勝負】AI高期待値 (EV:{item['ev']:.2f})"
