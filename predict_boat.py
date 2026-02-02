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

# ==========================================
# 🎯 戦略設定 (シミュレーション最適解)
# ==========================================
# 「数万通りの検証」で導き出された最強の設定値をそのまま採用します。
# びわこ(11)や住之江(12)の閾値が低いのは、検証の結果「それが一番儲かるから」です。

STRATEGY_DEFAULT = {'th': 0.070, 'k': 1} # リストにない会場用の暫定値

STRATEGY = {
    # 【関東】
    1:  {'th': 0.055, 'k': 1},  # 桐生 (101%)
    2:  {'th': 0.060, 'k': 3},  # 戸田 (190%)
    3:  {'th': 0.055, 'k': 5},  # 江戸川 (136%)
    5:  {'th': 0.070, 'k': 10}, # 多摩川 (119%)
    
    # 【東海】
    6:  {'th': 0.070, 'k': 2},  # 浜名湖 (130%)
    7:  {'th': 0.070, 'k': 1},  # 蒲郡 (243%)
    8:  {'th': 0.070, 'k': 8},  # 常滑 (103%)
    9:  {'th': 0.060, 'k': 3},  # 津 (138%)
    
    # 【北陸・近畿】
    10: {'th': 0.070, 'k': 10}, # 三国 (191%)
    11: {'th': 0.045, 'k': 1},  # びわこ (114%) ★高頻度ゾーン
    12: {'th': 0.050, 'k': 1},  # 住之江 (123%) ★高頻度ゾーン
    13: {'th': 0.065, 'k': 3},  # 尼崎 (111%)
    
    # 【四国・中国】
    15: {'th': 0.070, 'k': 1},  # 丸亀 (124%)
    16: {'th': 0.070, 'k': 1},  # 児島 (164%)
    18: {'th': 0.080, 'k': 1},  # 徳山 (298%) ★超厳選
    
    # 【九州】
    20: {'th': 0.075, 'k': 10}, # 若松 (126%)
    21: {'th': 0.065, 'k': 1},  # 芦屋 (119%)
    22: {'th': 0.065, 'k': 1},  # 福岡 (155%)
    23: {'th': 0.065, 'k': 1},  # 唐津 (138%)
    24: {'th': 0.065, 'k': 1},  # 大村 (124%)
}

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
    以下の{jcd}場のレースの「厳選買い目」を評価してください。
    AIの自信度に基づき、{len(bets_info)}点のセット買いを行います。
    
    [選手データ]
    {players_info}
    
    [買い目]
    {bets_text}
    
    【重要指示】
    各買い目について、オッズ妙味と展開を読み、**必ず【勝負】か【見送り】** で始めて、20文字以内でコメントしてください。
    
    出力例:
    1-2-3: 【勝負】 鉄板。銀行レース。
    1-4-2: 【勝負】 展開向けば万舟ある。
    """

    try:
        time.sleep(1.0)
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
                item['reason'] = f"【勝負】AIセット買い推奨 {ev_str}"
            else:
                item['reason'] = "【判断不能】オッズ不明"

# 安全な数値変換（これがないとシミュレーションと同じ計算ができない）
def to_float(val):
    try:
        if val is None or val == "": return 0.0
        return float(val)
    except:
        return 0.0

# ==========================================
# 🔮 予測ロジック (表の戦略を忠実に実行)
# ==========================================

def predict_race(raw, odds_data=None):
    model = load_model()
    
    jcd = raw.get('jcd', 0)
    wind = to_float(raw.get('wind', 0.0))
    rno = raw.get('rno', 0)
    
    # ここで表の数値を読み込む
    strat = STRATEGY.get(jcd, STRATEGY_DEFAULT)
    
    if strat['k'] == 0:
        return []

    # 1. データの強制数値化（バグ修正済みの処理）
    ex_list = []
    rows = []
    
    for i in range(1, 7):
        s = str(i)
        val_wr = to_float(raw.get(f'wr{s}', 0))
        val_mo = to_float(raw.get(f'mo{s}', 0))
        val_ex = to_float(raw.get(f'ex{s}', 0))
        val_st = to_float(raw.get(f'st{s}', 0.20))
        val_f  = to_float(raw.get(f'f{s}', 0))
        
        ex_list.append(val_ex)
        
        rows.append({
            'jcd': jcd, 
            'wind': wind, 
            'boat_no': i,
            'pid': raw.get(f'pid{s}', 0), 
            'wr': val_wr,
            'mo': val_mo, 
            'ex': val_ex,
            'st': val_st, 
            'f': val_f,
        })
    
    if sum(ex_list) == 0:
        return []

    df_race = pd.DataFrame(rows)

    # 2. Zスコア計算 (シミュレーションのロジックに合わせる)
    target_cols = ['wr', 'mo', 'ex', 'st']
    for col in target_cols:
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
    for i, j, k in permutations(range(6), 3):
        score = p1[i] * p2[j] * p3[k]
        combos.append({'combo': f"{b[i]}-{b[j]}-{b[k]}", 'score': score})
    
    combos.sort(key=lambda x: x['score'], reverse=True)
    best_bet = combos[0]
    
    best_score_pct = best_bet['score'] * 100
    threshold_pct = strat['th'] * 100
    
    # 3. 判定 (表で決まった閾値を適用)
    if best_bet['score'] < strat['th']:
        # 惜しい場合(基準の半分以上)はログに残す
        if best_bet['score'] > (strat['th'] * 0.5):
             print(f"ℹ️ [見送] {jcd}場{rno}R 1位:{best_bet['combo']} ({best_score_pct:.2f}%) / 基準:{threshold_pct:.1f}%")
        return []

    print(f"🔥 [勝負] {jcd}場{rno}R 条件クリア! スコア:{best_score_pct:.2f}% >= 基準:{threshold_pct:.1f}%")
    
    results = []
    # 表で決まった点数(k)だけセット買いリストに入れる
    for rank, item in enumerate(combos[:strat['k']]):
        results.append({
            'combo': item['combo'],
            'type': f"ランク{rank+1}",
            'profit': "計算中",
            'prob': f"{item['score']*100:.1f}",
            'roi': 0,
            'reason': "AI厳選戦略",
            'deadline': raw.get('deadline_time', '不明')
        })
        
    return results
