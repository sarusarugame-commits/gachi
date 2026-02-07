import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import joblib
from itertools import permutations

# ==========================================
# ⚙️ 設定: 攻めの穴狙い設定
# ==========================================

# --- 三連単 (3T) 攻めの設定 ---
# --- 三連単 (3T) 攻めの設定 ---
MIN_PROB_3T = 0.01        # 変更: 1.0% (大穴も拾う設定)
ODDS_CAP_3T = 80.0        # 変更: 80倍まで評価 (万舟狙い)
MAX_BETS_3T = 10          # 変更: 1レース最大10点 (シミュレーション通り手広く)
CONF_THRESH_3T = 0.15     # 変更: 15% (混戦レースも参加する)

STRATEGY_3T = {
    1: {'ev_thresh': 3.0},   # 桐生 (EV3.0以上)
    2: {'ev_thresh': 99.9},  # 戸田 (見送り)
    3: {'ev_thresh': 1.5},   # 江戸川 (EV1.5以上)
    4: {'ev_thresh': 1.5},   # 平和島 (EV1.5以上)
    5: {'ev_thresh': 99.9},  # 多摩川 (見送り)
    6: {'ev_thresh': 3.0},   # 浜名湖 (EV3.0以上)
    7: {'ev_thresh': 3.5},   # 蒲郡 (EV3.5以上)
    8: {'ev_thresh': 4.0},   # 常滑 (EV4.0以上)
    9: {'ev_thresh': 99.9},  # 津 (見送り)
    10: {'ev_thresh': 99.9}, # 三国 (見送り)
    11: {'ev_thresh': 3.5},  # びわこ (EV3.5以上)
    12: {'ev_thresh': 99.9}, # 住之江 (見送り)
    13: {'ev_thresh': 99.9}, # 尼崎 (見送り)
    14: {'ev_thresh': 3.5},  # 鳴門 (EV3.5以上)
    15: {'ev_thresh': 99.9}, # 丸亀 (見送り)
    16: {'ev_thresh': 3.5},  # 児島 (EV3.5以上)
    17: {'ev_thresh': 3.5},  # 宮島 (EV3.5以上)
    18: {'ev_thresh': 2.5},  # 徳山 (EV2.5以上)
    19: {'ev_thresh': 4.0},  # 下関 (EV4.0以上)
    20: {'ev_thresh': 99.9}, # 若松 (見送り)
    21: {'ev_thresh': 99.9}, # 芦屋 (見送り)
    22: {'ev_thresh': 3.5},  # 福岡 (EV3.5以上)
    23: {'ev_thresh': 99.9}, # 唐津 (見送り)
    24: {'ev_thresh': 3.0},  # 大村 (EV3.0以上)
}          

# --- 二連単 (2T) 設定 ---
# --- 二連単 (2T) 設定 ---
# ⚠️ シミュレーション結果「JCD 8, 10, 16, 21 のみ」を反映
# 見送りの会場は閾値を「99.9」にして物理的に買わせないようにする

MIN_PROB_2T = 0.01  # そのままでOK（EVで弾かれるため）
ODDS_CAP_2T = 100.0
MAX_BETS_2T = 8     # 厳選されるので8のままでも良いが、念のため減らしてもOK
CONF_THRESH_2T = 0.0 # モデルの確率自体は使うので0.0でOK

STRATEGY_2T = {
    1: {'ev_thresh': 99.9},  # 桐生 (見送り)
    2: {'ev_thresh': 99.9},  # 戸田 (見送り)
    3: {'ev_thresh': 99.9},  # 江戸川 (見送り)
    4: {'ev_thresh': 99.9},  # 平和島 (見送り)
    5: {'ev_thresh': 99.9},  # 多摩川 (見送り)
    6: {'ev_thresh': 99.9},  # 浜名湖 (見送り)
    7: {'ev_thresh': 99.9},  # 蒲郡 (見送り)
    8: {'ev_thresh': 4.0},   # 常滑 (★EV 4.0以上)
    9: {'ev_thresh': 99.9},  # 津 (見送り)
    10: {'ev_thresh': 4.0},  # 三国 (★EV 4.0以上)
    11: {'ev_thresh': 99.9}, # びわこ (見送り)
    12: {'ev_thresh': 99.9}, # 住之江 (見送り)
    13: {'ev_thresh': 99.9}, # 尼崎 (見送り)
    14: {'ev_thresh': 99.9}, # 鳴門 (見送り)
    15: {'ev_thresh': 99.9}, # 丸亀 (見送り)
    16: {'ev_thresh': 3.0},  # 児島 (★EV 3.0以上)
    17: {'ev_thresh': 99.9}, # 宮島 (見送り)
    18: {'ev_thresh': 99.9}, # 徳山 (見送り)
    19: {'ev_thresh': 99.9}, # 下関 (見送り)
    20: {'ev_thresh': 99.9}, # 若松 (見送り)
    21: {'ev_thresh': 2.5},  # 芦屋 (★EV 2.5以上)
    22: {'ev_thresh': 99.9}, # 福岡 (見送り)
    23: {'ev_thresh': 99.9}, # 唐津 (見送り)
    24: {'ev_thresh': 99.9}, # 大村 (見送り)
}

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
    if not OPENAI_AVAILABLE:
        print("⚠️ Groq Error: 'openai' module not found. pip install openai")
        return None
    if _GROQ_CLIENT is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("⚠️ Groq Error: GROQ_API_KEY env var not found.")
            return None
        try:
            _GROQ_CLIENT = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=api_key,
                max_retries=3, 
                timeout=20.0
            )
        except Exception as e:
            print(f"⚠️ Groq Init Error: {e}")
            return None
    return _GROQ_CLIENT

def check_groq_setup():
    """起動時にGroqの設定を確認する"""
    print("🤖 Groqセットアップ確認中...")
    if not OPENAI_AVAILABLE:
        print("❌ 'openai' ライブラリが見つかりません。")
        return
    
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("❌ 環境変数 GROQ_API_KEY が設定されていません。")
        return
    
    client = get_groq_client()
    if client:
        print("✅ Groqクライアント初期化成功")
    else:
        print("❌ Groqクライアント初期化失敗")

# ==========================================
# 📂 モデル管理 (2T:単一ファイル, 3T:一括pkl)
# ==========================================
MODELS_3T = None # 会場別辞書
MODEL_2T = None  # 単一モデル

FILE_3T = "boatrace_models_all.pkl"
FILE_2T = "boatrace_model_2t.txt"

def load_models():
    """起動時に2つのモデルを読み込む"""
    global MODELS_3T, MODEL_2T
    
    # --- 3連単 (会場別pkl) ---
    if MODELS_3T is None:
        if os.path.exists(FILE_3T):
            try:
                print(f"📂 3Tモデル読み込み中: {FILE_3T}")
                MODELS_3T = joblib.load(FILE_3T)
                print("✅ 3Tモデル読み込み完了")
            except Exception as e:
                print(f"❌ 3Tモデル読み込みエラー: {e}")
                MODELS_3T = {}
        else:
            print(f"⚠️ 3Tモデルなし: {FILE_3T}")
            MODELS_3T = {}

    # --- 2連単 (全体txt) ---
    if MODEL_2T is None:
        if os.path.exists(FILE_2T):
            try:
                print(f"📂 2Tモデル読み込み中: {FILE_2T}")
                MODEL_2T = lgb.Booster(model_file=FILE_2T)
                print("✅ 2Tモデル読み込み完了")
            except Exception as e:
                print(f"❌ 2Tモデル読み込みエラー: {e}")
                MODEL_2T = None
        else:
            print(f"⚠️ 2Tモデルなし: {FILE_2T}")
            MODEL_2T = None

def get_3t_model(jcd):
    global MODELS_3T
    if MODELS_3T is None: load_models()
    return MODELS_3T.get(jcd)

def get_2t_model():
    global MODEL_2T
    if MODEL_2T is None: load_models()
    return MODEL_2T

def to_float(val):
    try:
        if val is None or val == "": return 0.0
        return float(val)
    except: return 0.0

# ==========================================
# 🔮 1. 候補出し (2T & 3T対応)
# ==========================================
def predict_race(raw):
    jcd = int(raw.get('jcd', 0))
    
    # データフレーム作成
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
    # Zスコア計算
    for col in ['wr', 'mo', 'ex', 'st']:
        m, s = df[col].mean(), df[col].std()
        df[f'{col}_z'] = (df[col] - m) / (s if s != 0 else 1e-6)

    df['pid'] = df['pid'].astype('category')
    
    # 特徴量リスト (学習時と合わせる)
    features = ['boat_no', 'pid', 'wind', 'wr', 'mo', 'ex', 'st', 'f', 'wr_z', 'mo_z', 'ex_z', 'st_z']
    
    candidates = []
    b = df['boat_no'].values
    
    # ----------------------------------------
    # 🎯 3連単予測 (会場別モデル)
    # ----------------------------------------
    max_p1 = 0.0
    max_removed_prob = 0.0
    
    model_3t = get_3t_model(jcd)
    if model_3t:
        try:
            # 3T用予測 (特徴量からjcdを除外したもので学習している前提)
            p = model_3t.predict(df[features])
            p1, p2, p3 = p[:, 0], p[:, 1], p[:, 2] 
            
            max_p1 = max(p1)
            
            if max_p1 >= CONF_THRESH_3T:
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
            print(f"⚠️ 3T予測エラー JCD{jcd}: {e}")

    # ----------------------------------------
    # 🎯 2連単予測 (全体モデル)
    # ----------------------------------------
    model_2t = get_2t_model()
    if model_2t:
        try:
            # 2T用特徴量 (jcdを含める)
            df_2t = df.copy()
            df_2t['jcd'] = jcd
            df_2t['jcd'] = df_2t['jcd'].astype('category')
            
            # 全体モデルは jcd を含む特徴量で学習している
            features_2t = ['jcd'] + features
            
            p_2t = model_2t.predict(df_2t[features_2t])
            # 多クラス分類 (0=1着, 1=2着...)
            p1_2t, p2_2t = p_2t[:, 0], p_2t[:, 1]
            
            for i, j in permutations(range(6), 2):
                prob = p1_2t[i] * p2_2t[j]
                
                if prob >= MIN_PROB_2T:
                    candidates.append({
                        'combo': f"{b[i]}-{b[j]}", 
                        'raw_prob': prob, 
                        'prob': round(prob * 100, 1),
                        'type': '2t'
                    })
        except Exception as e:
            print(f"⚠️ 2T予測エラー JCD{jcd}: {e}")

    if not candidates:
        return [], 0.0, 0.0, True # 何も出なくてもエラーではない

    candidates.sort(key=lambda x: x['raw_prob'], reverse=True)
    return candidates, max_p1, max_removed_prob, True

# ==========================================
# 💰 2. EVフィルタ
# ==========================================
def filter_and_sort_bets(candidates, odds_2t, odds_3t, jcd):
    final_bets = []
    max_ev = 0.0
    
    # 戦略設定（2t, 3tで分けるならここ）
    # 今回は簡易的に共通閾値だが、本来は辞書等で分ける
    
    for c in candidates:
        combo = c['combo']
        prob = c['raw_prob']
        bet_type = c['type']
        
        real_o = 0.0
        cap = 100.0
        thresh = 1.0 # デフォルト
        
        if bet_type == '3t':
            real_o = odds_3t.get(combo, 0.0)
            cap = ODDS_CAP_3T
            # ★会場ごとの設定(ev_thresh)を読み込む
            thresh = STRATEGY_3T.get(jcd, {}).get('ev_thresh', 99.9)
        elif bet_type == '2t':
            real_o = odds_2t.get(combo, 0.0)
            cap = ODDS_CAP_2T
            # 修正: 会場ごとのEV設定を適用
            thresh = STRATEGY_2T.get(jcd, {}).get('ev_thresh', 99.9)

        if real_o > 0:
            ev = prob * min(real_o, cap)
            if ev > max_ev: max_ev = ev
            
            if ev >= thresh:
                c.update({'odds': real_o, 'ev': ev})
                final_bets.append(c)
    
    # 賭け式ごとに購入数制限をかける処理が必要ならここに追加
    # 今は単純にEV順で上位を返す
    final_bets.sort(key=lambda x: x['ev'], reverse=True)
    
    # 3連単と2連単が混ざると見にくいので、上位からつまむが
    # それぞれ MAX_BETS まで取得するようにする
    
    bets_3t = [b for b in final_bets if b['type'] == '3t'][:MAX_BETS_3T]
    bets_2t = [b for b in final_bets if b['type'] == '2t'][:MAX_BETS_2T]
    
    merged = bets_3t + bets_2t
    merged.sort(key=lambda x: x['ev'], reverse=True)
    
    return merged, max_ev, 0.0

# ==========================================
# 📝 3. 解説生成 (変更なし)
# ==========================================
def generate_batch_reasons(jcd, bets_info, raw_data):
    client = get_groq_client()
    if not client: return {}
    
    players_info = ""
    for i in range(1, 7):
        players_info += f"{i}号艇:勝率{raw_data.get(f'wr{i}',0)} "

    bets_text = ""
    for b in bets_info:
        bets_text += f"- {b['combo']}: 確率{b['prob']}% オッズ{b['odds']} (EV:{b['ev']:.2f})\n"

    prompt = f"""
    ボートレース予想家として、以下の{jcd}場の買い目を解説せよ。
    [選手] {players_info}
    [買い目] {bets_text}
    【指示】
    各買い目について、なぜチャンスなのか 300文字以内 でコメント。
    「穴狙い」の視点を入れて解説すること。

    【出力形式】
    必ず以下の形式で1行につき1つの買い目の解説を出力すること。余計な挨拶は不要。
    買い目: 解説文
    
    例:
    1-2-3: 1号艇の逃げ信頼だが2号艇の差しも警戒...
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
        return comments
    except Exception as e:
        print(f"⚠️ Groq API Error: {e}")
        return {}

def attach_reason(results, raw, odds_map=None):
    if not results: return
    jcd = raw.get('jcd', 0)
    # 解説生成（コスト節約のため、上位3つくらいに絞っても良い）
    ai_comments = generate_batch_reasons(jcd, results[:5], raw)
    for item in results:
        ai_msg = ai_comments.get(item['combo'])
        if ai_msg:
            item['reason'] = f"{ai_msg} (EV:{item['ev']:.2f})"
        else:
            item['reason'] = f"【勝負】AI推奨 (EV:{item['ev']:.2f})"
