import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import joblib # ★追加
from itertools import permutations

# ==========================================
# ⚙️ 設定: 攻めの穴狙い設定 (ROI 143% Ver)
# ==========================================

# --- 三連単 (3T) 攻めの設定 ---
MIN_PROB_3T = 0.01        
ODDS_CAP_3T = 80.0        
MAX_BETS_3T = 10          
CONF_THRESH_3T = 0.15     
STRATEGY_3T = {}          

# --- 二連単 (2T) 設定 ---
MIN_PROB_2T = 0.01
ODDS_CAP_2T = 100.0
MAX_BETS_2T = 8
CONF_THRESH_2T = 0.0
STRATEGY_2T = {}

# ==========================================
# 🤖 Groq 設定
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

# --- モデル管理 (一括ロード方式) ---
ALL_MODELS = None
MODEL_FILE = "boatrace_models_all.pkl" # ★まとめたファイル名

def load_model():
    """起動時に一括ファイルを読み込む"""
    global ALL_MODELS
    if ALL_MODELS is None:
        if os.path.exists(MODEL_FILE):
            try:
                print(f"📂 モデル読み込み中: {MODEL_FILE}")
                ALL_MODELS = joblib.load(MODEL_FILE)
                print("✅ モデル読み込み完了")
            except Exception as e:
                print(f"❌ モデル読み込みエラー: {e}")
                ALL_MODELS = {}
        else:
            print("⚠️ モデルファイルが見つかりません")
            ALL_MODELS = {}

def get_model_for_jcd(jcd):
    """メモリ上の辞書から会場モデルを返す"""
    global ALL_MODELS
    if ALL_MODELS is None:
        load_model()
    
    if ALL_MODELS and jcd in ALL_MODELS:
        return ALL_MODELS[jcd]
    return None

def to_float(val):
    try:
        if val is None or val == "": return 0.0
        return float(val)
    except: return 0.0

# ==========================================
# 🔮 1. 候補出し (辞書型モデル対応)
# ==========================================
def predict_race(raw):
    jcd = int(raw.get('jcd', 0))
    
    # モデル取得
    model = get_model_for_jcd(jcd)
    if model is None:
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

    df['pid'] = df['pid'].astype('category')
    
    # 学習時と同じ特徴量 (jcd除外)
    features = ['boat_no', 'pid', 'wind', 'wr', 'mo', 'ex', 'st', 'f', 'wr_z', 'mo_z', 'ex_z', 'st_z']
    
    candidates = []
    max_p1 = 0.0
    max_removed_prob = 0.0
    b = df['boat_no'].values

    try:
        p = model.predict(df[features])
        p1, p2, p3 = p[:, 0], p[:, 1], p[:, 2] 
        
        current_max = max(p1)
        max_p1 = max(max_p1, current_max)
        
        if current_max >= CONF_THRESH_3T:
            for i, j, k in permutations(range(6), 3):
                prob = p1[i] * p2[j] * p3[k]
                if prob > max_removed_prob: max_removed_prob = prob
                
                if prob >= MIN_PROB_3T:
                    candidates.append({
                        'combo': f"{b[i]}-{b[j]}-{b[k]}", 
                        'raw_prob': prob, 
                        'prob': round(prob * 100, 1),
                        'type': '3t'
                    })
    except Exception as e:
        print(f"Prediction Error JCD{jcd}: {e}")
        return [], 0.0, 0.0, False

    candidates.sort(key=lambda x: x['raw_prob'], reverse=True)
    return candidates, max_p1, max_removed_prob, True

# ==========================================
# 💰 2. EVフィルタ
# ==========================================
def filter_and_sort_bets(candidates, odds_2t, odds_3t, jcd):
    final_bets = []
    max_ev = 0.0
    strategy_thresh = 1.5 
    
    for c in candidates:
        combo = c['combo']
        prob = c['raw_prob']
        ev = 0.0
        
        if c['type'] == '3t':
            real_o = odds_3t.get(combo, 0.0)
            if real_o > 0:
                ev = prob * min(real_o, ODDS_CAP_3T)
                if ev > max_ev: max_ev = ev
                if ev >= strategy_thresh:
                    c.update({'odds': real_o, 'ev': ev})
                    final_bets.append(c)
    
    final_bets.sort(key=lambda x: x['ev'], reverse=True)
    return final_bets[:MAX_BETS_3T], max_ev, strategy_thresh

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
        bets_text += f"- {b['combo']}: 確率{b['prob']}% オッズ{b['odds']} (期待値{b['ev']:.2f})\n"

    prompt = f"""
    ボートレース予想家として、以下の{jcd}場の買い目を解説せよ。
    [選手] {players_info}
    [買い目] {bets_text}
    【指示】
    各買い目について、なぜチャンスなのか 300文字以内 でコメント。
    「穴狙い」の視点を入れて解説すること。
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
            item['reason'] = f"【勝負】AI穴推奨 (EV:{item['ev']:.2f})"
