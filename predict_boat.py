import pandas as pd
import numpy as np
import lightgbm as lgb
import os
from itertools import permutations

# ==========================================
# ⚙️ 設定: 券種別・完全独立パラメータ
# ==========================================

# --- 三連単 (3T) 黄金律設定 ---
MIN_PROB_3T = 0.03
ODDS_CAP_3T = 40.0
MAX_BETS_3T = 6
CONF_THRESH_3T = 0.20
STRATEGY_3T = {
    2: 2.0, 3: 1.2, 5: 2.0, 6: 1.6, 8: 1.8, 9: 1.4, 10: 1.3,
    11: 2.5, 13: 1.6, 14: 1.6, 16: 1.5, 19: 1.3, 20: 2.0,
    22: 1.2, 23: 1.5, 24: 1.5
}

# --- 二連単 (2T) ROI 130% 厳選設定 ---
MIN_PROB_2T = 0.01
ODDS_CAP_2T = 100.0
MAX_BETS_2T = 8
CONF_THRESH_2T = 0.0
STRATEGY_2T = {
    8: 4.0, 10: 4.0, 16: 3.0, 21: 2.5
}

# ==========================================
# 🤖 Groq (OpenAI Client Wrapper) 設定
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
                max_retries=3, 
                timeout=20.0
            )
        except: return None
    return _GROQ_CLIENT

# --- モデル管理 ---
MODELS = {'3t': None, '2t': None}

def load_model():
    # 3Tモデル
    if MODELS['3t'] is None:
        if os.path.exists("boatrace_model.txt"):
            MODELS['3t'] = lgb.Booster(model_file="boatrace_model.txt")
        elif os.path.exists("boat_race_model_3t.txt"):
            MODELS['3t'] = lgb.Booster(model_file="boat_race_model_3t.txt")
    
    # 2Tモデル
    if MODELS['2t'] is None:
        if os.path.exists("boatrace_model_2t.txt"):
            MODELS['2t'] = lgb.Booster(model_file="boatrace_model_2t.txt")
        
    return MODELS

def to_float(val):
    try:
        if val is None or val == "": return 0.0
        return float(val)
    except: return 0.0

# ==========================================
# 🔮 1. 候補出し (3T / 2T 独立判定)
# ==========================================
def predict_race(raw):
    """
    戻り値: (候補リスト, 最大自信度, 戦略対象フラグ)
    """
    load_model()
    jcd = int(raw.get('jcd', 0))
    use_3t = jcd in STRATEGY_3T
    use_2t = jcd in STRATEGY_2T
    
    if not use_3t and not use_2t:
        return [], 0.0, 0.0, False

    # 特徴量生成
    rows = []
    ex_list = []
    wind = to_float(raw.get('wind', 0.0))
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
    
    if sum(ex_list) == 0: return [], 0.0, 0.0, True

    df = pd.DataFrame(rows)
    for col in ['wr', 'mo', 'ex', 'st']:
        m, s = df[col].mean(), df[col].std()
        df[f'{col}_z'] = (df[col] - m) / (s if s != 0 else 1e-6)

    df['jcd'] = df['jcd'].astype('category')
    df['pid'] = df['pid'].astype('category')
    features = ['jcd', 'boat_no', 'pid', 'wind', 'wr', 'mo', 'ex', 'st', 'f', 'wr_z', 'mo_z', 'ex_z', 'st_z']
    
    candidates = []
    max_p1 = 0.0
    max_removed_prob = 0.0
    b = df['boat_no'].values

    # --- 三連単 判定 ---
    if MODELS['3t'] and use_3t:
        p = MODELS['3t'].predict(df[features])
        p1, p2, p3 = p[:, 0], p[:, 1], p[:, 2]
        current_max = max(p1)
        max_p1 = max(max_p1, current_max)
        
        if current_max >= CONF_THRESH_3T:
            for i, j, k in permutations(range(6), 3):
                prob = p1[i] * p2[j] * p3[k]
                if prob > max_removed_prob: max_removed_prob = prob # 棄却された最大確率を記録
                
                if prob >= MIN_PROB_3T:
                    candidates.append({
                        'combo': f"{b[i]}-{b[j]}-{b[k]}", 
                        'raw_prob': prob, 
                        'prob': round(prob * 100, 1),
                        'type': '3t'
                    })

    # --- 二連単 判定 ---
    if MODELS['2t'] and use_2t:
        p_2t = MODELS['2t'].predict(df[features])
        p1_2, p2_2 = p_2t[:, 0], p_2t[:, 1]
        current_max = max(p1_2)
        max_p1 = max(max_p1, current_max)

        if current_max >= CONF_THRESH_2T:
            for i, j in permutations(range(6), 2):
                prob = p1_2[i] * p2_2[j]
                if prob > max_removed_prob: max_removed_prob = prob

                if prob >= MIN_PROB_2T:
                    candidates.append({
                        'combo': f"{b[i]}-{b[j]}", 
                        'raw_prob': prob, 
                        'prob': round(prob * 100, 1),
                        'type': '2t'
                    })

    # 確率順にソート (EV計算前の一時ソート)
    candidates.sort(key=lambda x: x['raw_prob'], reverse=True)
    return candidates, max_p1, max_removed_prob, True

# ==========================================
# 💰 2. EVフィルタ
# ==========================================
def filter_and_sort_bets(candidates, odds_2t, odds_3t, jcd):
    final_2t, final_3t = [], []
    max_ev = 0.0
    
    # 戦略閾値の取得 (3T優先、なければ2T。ログ用)
    strategy_thresh = STRATEGY_3T.get(jcd) if jcd in STRATEGY_3T else STRATEGY_2T.get(jcd, 99.0)

    for c in candidates:
        combo = c['combo']
        prob = c['raw_prob']
        ev = 0.0
        
        if c['type'] == '2t':
            real_o = odds_2t.get(combo, 0.0)
            if real_o > 0:
                ev = prob * min(real_o, ODDS_CAP_2T)
                if ev > max_ev: max_ev = ev
                if ev >= STRATEGY_2T.get(jcd, 99.0):
                    c.update({'odds': real_o, 'ev': ev})
                    final_2t.append(c)
        else:
            real_o = odds_3t.get(combo, 0.0)
            if real_o > 0:
                ev = prob * min(real_o, ODDS_CAP_3T)
                if ev > max_ev: max_ev = ev
                if ev >= STRATEGY_3T.get(jcd, 99.0):
                    c.update({'odds': real_o, 'ev': ev})
                    final_3t.append(c)
    
    # 修正: 期待値(EV)が高い順にソートし直す
    final_2t.sort(key=lambda x: x['ev'], reverse=True)
    final_3t.sort(key=lambda x: x['ev'], reverse=True)
            
    return final_2t[:MAX_BETS_2T] + final_3t[:MAX_BETS_3T], max_ev, strategy_thresh

# ==========================================
# 📝 3. 解説生成
# ==========================================
def generate_batch_reasons(jcd, bets_info, raw_data):
    client = get_groq_client()
    if not client: return {}
    
    players_info = ""
    for i in range(1, 7):
        players_info += f"{i}号艇:勝率{raw_data.get(f'wr{i}',0)} "

    bets_text = ""
    for b in bets_info:
        bets_text += f"- {b['combo']}({b['type'].upper()}): 確率{b['prob']}% オッズ{b['odds']} (期待値{b['ev']:.2f})\n"

    prompt = f"""
    ボートレース予想家として、以下の{jcd}場の買い目を解説せよ。
    [選手] {players_info}
    [買い目] {bets_text}
    【指示】
    各買い目について、なぜチャンスなのか 300文字以内 でコメント。
    必ず 【勝負】 か 【見送り】 で始めること。
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

def attach_reason(results, raw, odds_map=None):
    if not results: return
    jcd = raw.get('jcd', 0)
    ai_comments = generate_batch_reasons(jcd, results, raw)
    for item in results:
        ai_msg = ai_comments.get(item['combo'])
        if ai_msg:
            item['reason'] = f"{ai_msg} (EV:{item['ev']:.2f})"
        else:
            item['reason'] = f"【勝負】AI推奨 (EV:{item['ev']:.2f})"
