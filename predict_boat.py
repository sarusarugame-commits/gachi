import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import zipfile
import time
import random
from itertools import permutations
import json

# ==========================================
# 🤖 AI解説機能 (Groq / OpenAI) の設定
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

# ==========================================
# ⚙️ 最強設定 (回収率130%モデル)
# ==========================================
MODEL_FILE = "boatrace_model.txt"  # 新しいモデルファイル名

# 基本フィルタ設定
MIN_PROB_THRESHOLD = 0.03       # 確率3%以上のみ対象
MAX_BETS_PER_RACE = 6           # 1レース最大6点
CALC_ODDS_CAP = 40.0            # オッズキャップ40倍
RACE_CONFIDENCE_THRESHOLD = 0.20 # レース自信度20%

# 会場ごとの最適EV閾値 (2023-2025年の分析結果)
# 99.9 は「見送り」設定
BEST_EV_THRESHOLDS = {
    1: 1.4,  2: 1.8,  3: 99.9, 4: 1.2,  5: 99.9, 6: 1.5,
    7: 1.8,  8: 99.9, 9: 1.3,  10: 1.8, 11: 2.0, 12: 99.9,
    13: 2.0, 14: 1.4, 15: 1.8, 16: 1.8, 17: 2.0, 18: 2.0,
    19: 1.6, 20: 1.8, 21: 1.4, 22: 1.4, 23: 1.3, 24: 99.9
}

AI_MODEL = None

def load_model():
    global AI_MODEL
    if AI_MODEL is None:
        # 新しいモデルを優先、なければ旧モデルを探す
        if os.path.exists(MODEL_FILE):
            print(f"📂 モデルファイルを検出: {MODEL_FILE}")
            AI_MODEL = lgb.Booster(model_file=MODEL_FILE)
        elif os.path.exists("boat_race_model_3t.txt"):
            print(f"📂 モデルファイルを検出(旧名): boat_race_model_3t.txt")
            AI_MODEL = lgb.Booster(model_file="boat_race_model_3t.txt")
        else:
            raise FileNotFoundError(f"モデルファイルが見つかりません。")
    return AI_MODEL

def to_float(val):
    try:
        if val is None or val == "": return 0.0
        return float(val)
    except:
        return 0.0

# ==========================================
# 🔮 予測ロジック (緩めに候補を出す)
# ==========================================
def predict_race(raw):
    """
    確率計算を行い、基準(3%)を超える候補を「広めに」返す。
    最終的な絞り込み(EVフィルタ)は main.py 側で行う。
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
            'jcd': jcd, 
            'wind': wind, 
            'boat_no': i,
            'pid': raw.get(f'pid{s}', 0), 
            'wr': to_float(raw.get(f'wr{s}', 0)),
            'mo': to_float(raw.get(f'mo{s}', 0)), 
            'ex': val_ex,
            'st': to_float(raw.get(f'st{s}', 0.20)), 
            'f': to_float(raw.get(f'f{s}', 0)),
        })
    
    if sum(ex_list) == 0: return []

    df_race = pd.DataFrame(rows)

    # Zスコア計算
    for col in ['wr', 'mo', 'ex', 'st']:
        mean_val = df_race[col].mean()
        std_val = df_race[col].std()
        if std_val == 0: std_val = 1e-6
        df_race[f'{col}_z'] = (df_race[col] - mean_val) / std_val

    df_race['jcd'] = df_race['jcd'].astype('category')
    df_race['pid'] = df_race['pid'].astype('category')
    
    features = [
        'jcd', 'boat_no', 'pid', 'wind',
        'wr', 'mo', 'ex', 'st', 'f',
        'wr_z', 'mo_z', 'ex_z', 'st_z'
    ]

    try:
        preds = model.predict(df_race[features])
        p1, p2, p3 = preds[:, 0], preds[:, 1], preds[:, 2]
    except Exception as e:
        print(f"❌ 予測エラー: {e}")
        return []

    b = df_race['boat_no'].values
    combos = []
    
    # 自信度チェック
    max_win_prob = max(p1)
    if max_win_prob < RACE_CONFIDENCE_THRESHOLD:
        return [] # 自信がないレースは見送り

    for i, j, k in permutations(range(6), 3):
        score = p1[i] * p2[j] * p3[k]
        
        # 確率3%以上のみ候補にする
        if score >= MIN_PROB_THRESHOLD:
            combos.append({
                'combo': f"{b[i]}-{b[j]}-{b[k]}", 
                'prob': round(score * 100, 1), 
                'raw_prob': score
            })
    
    # 確率高い順にソートして、上位20件くらいを返す(EV計算用)
    combos.sort(key=lambda x: x['raw_prob'], reverse=True)
    return combos[:20]

# ==========================================
# 💰 EVフィルタリング (最強の肝)
# ==========================================
def filter_and_sort_bets(candidates, odds_map, jcd):
    """
    候補リストに対し、オッズを適用してEV計算 -> 閾値チェック -> 厳選を行う
    """
    threshold = BEST_EV_THRESHOLDS.get(jcd, 99.9)
    if threshold >= 99.0:
        return [] # 見送り会場

    final_bets = []
    
    for bet in candidates:
        combo = bet['combo']
        prob = bet['raw_prob'] # 0.03 etc
        
        # オッズ取得
        real_odds = odds_map.get(combo, 0.0)
        if real_odds == 0: continue
        
        # オッズキャップ適用
        calc_odds = min(real_odds, CALC_ODDS_CAP)
        
        # 期待値計算
        ev = prob * calc_odds
        
        # 閾値チェック
        if ev >= threshold:
            bet['odds'] = real_odds
            bet['ev'] = ev
            # reasonは後でGroqで上書きされるが、念のため入れておく
            bet['reason'] = f"EV:{ev:.2f} (基準{threshold})" 
            final_bets.append(bet)
            
    # EVが高い順にソート
    final_bets.sort(key=lambda x: x['ev'], reverse=True)
    
    # 上位N点に絞る
    return final_bets[:MAX_BETS_PER_RACE]

# ==========================================
# 📝 解説生成 (Groq復活！)
# ==========================================
def generate_batch_reasons(jcd, bets_info, raw_data):
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
    あなたはボートレース初心者にも優しく分かりやすく解説するベテラン予想家です。
    以下の{jcd}場のレースでAIが選んだ「推奨買い目」について、
    なぜその買い目がチャンスなのか、初心者でも納得できる理由をコメントしてください。
    
    [選手データ]
    {players_info}
    
    [買い目]
    {bets_text}
    
    【重要指示】
    1. 専門用語はなるべく使わず、平易な言葉で説明してください。
    2. 「期待値が高い」「確率が高い」といった根拠も交えてください。
    3. 各買い目に対し、必ず **【勝負】** または **【見送り】** で始めて、30文字以内でコメントしてください。
    
    出力例:
    1-2-3: 【勝負】 1番の実力が圧倒的！安心して見ていられます。
    """

    try:
        time.sleep(1.0)
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "あなたは親切で分かりやすいボートレース解説者です。"},
                {"role": "user", "content": prompt}
            ],
            model=selected_model, 
            temperature=0.7,
            max_tokens=400,
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
    """
    買い目リストに対して、Groqを使って解説文を付与する
    """
    if not results: return
    
    jcd = raw.get('jcd', 0)
    
    # Groqに投げるためのデータ整形
    bets_to_analyze = []
    for item in results:
        bets_to_analyze.append({
            'combo': item['combo'], 
            'prob': item['prob'], # %表記
            'odds': item.get('odds'), 
            'ev': item.get('ev')
        })

    # Groqで解説生成
    ai_comments = generate_batch_reasons(jcd, bets_to_analyze, raw)
    
    # 結果に反映
    for item in results:
        combo = item['combo']
        ev_val = item.get('ev')
        ai_comment = ai_comments.get(combo)
        
        ev_str = f"(EV:{ev_val:.2f})" if ev_val else ""
        
        if ai_comment:
            item['reason'] = f"{ai_comment} {ev_str}"
        else:
            if ev_val:
                item['reason'] = f"【勝負】AI高期待値の狙い目 {ev_str}"
            else:
                item['reason'] = "【判断不能】解説生成失敗"
