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

# ★★★ 会場別・最強戦略ポートフォリオ ★★★
# リストにない会場（負け越している会場）は「見送り (th=1.0)」にして鉄壁の防御を固めます。
STRATEGY_DEFAULT = {'th': 1.0, 'k': 0} 

STRATEGY = {
    # 【関東】
    1:  {'th': 0.065, 'k': 1}, # 桐生: 厳選1点 (回収率218%)
    2:  {'th': 0.050, 'k': 5}, # 戸田: 5点流し (回収率105%)
    3:  {'th': 0.060, 'k': 8}, # 江戸川: 荒れるので8点 (回収率144%)
    4:  {'th': 0.050, 'k': 5}, # 平和島: 5点 (回収率104%)
    5:  {'th': 0.040, 'k': 1}, # 多摩川: 1点 (回収率109%)
    
    # 【東海】
    7:  {'th': 0.065, 'k': 1}, # 蒲郡: 1点 (回収率111%)
    8:  {'th': 0.070, 'k': 5}, # 常滑: 5点 (回収率136%)
    9:  {'th': 0.055, 'k': 1}, # 津: 1点 (回収率153%)
    10: {'th': 0.060, 'k': 8}, # 三国: 8点 (回収率162%)
    
    # 【近畿・四国】
    11: {'th': 0.045, 'k': 1}, # びわこ: 1点 (回収率106%)
    12: {'th': 0.060, 'k': 1}, # 住之江: 1点 (回収率109%)
    13: {'th': 0.040, 'k': 1}, # 尼崎: 1点 (回収率103%)
    15: {'th': 0.065, 'k': 1}, # 丸亀: 1点 (回収率268%!)
    16: {'th': 0.055, 'k': 1}, # 児島: 1点 (回収率155%)
    18: {'th': 0.070, 'k': 1}, # 徳山: 1点 (回収率315%!!)
    19: {'th': 0.065, 'k': 1}, # 下関: 1点 (回収率139%)
    20: {'th': 0.070, 'k': 8}, # 若松: 8点 (回収率151%)
    
    # 【九州】
    21: {'th': 0.060, 'k': 1}, # 芦屋: 1点 (回収率106%)
    22: {'th': 0.055, 'k': 1}, # 福岡: 1点 (回収率111%)
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
    この会場の最適戦略に従い選出された買い目です。
    
    [選手データ]
    {players_info}
    
    [買い目]
    {bets_text}
    
    【重要指示】
    各買い目について、自信度を考慮し、**必ず【勝負】か【見送り】** で始めて、20文字以内でコメントしてください。
    
    出力例:
    1-2-3: 【勝負】 鉄板データ。
    1-4-2: 【勝負】 妙味あり。
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
                item['reason'] = f"【勝負】最適戦略適合 {ev_str}"
            else:
                item['reason'] = "【判断不能】オッズ不明"

def predict_race(raw, odds_data=None):
    model = load_model()
    
    jcd = raw.get('jcd', 0)
    wind = raw.get('wind', 0.0)
    rno = raw.get('rno', 0)
    
    # ★ 会場ごとの最適戦略を取得 (ない会場はデフォルト=見送り)
    strat = STRATEGY.get(jcd, STRATEGY_DEFAULT)
    
    # 戦略が「見送り(k=0)」なら即終了
    if strat['k'] == 0:
        return []

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

    # 偏差値(Z-score)計算
    target_cols = ['wr', 'mo', 'ex', 'st']
    for col in target_cols:
        mean_val = df_race[col].mean()
        std_val = df_race[col].std()
        df_race[f'{col}_z'] = (df_race[col] - mean_val) / (std_val + 1e-6)

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
        print(f"Prediction Error: {e}")
        return []

    b = df_race['boat_no'].values
    combos = []
    for i, j, k in permutations(range(6), 3):
        score = p1[i] * p2[j] * p3[k]
        combos.append({'combo': f"{b[i]}-{b[j]}-{b[k]}", 'score': score})
    combos.sort(key=lambda x: x['score'], reverse=True)
    
    results = []
    # k点まで取得 (ただし閾値以下の買い目は捨てる)
    for rank, item in enumerate(combos[:strat['k']]):
        if item['score'] < strat['th']:
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
