import pandas as pd
import numpy as np
import lightgbm as lgb
import os
from itertools import permutations

# ==========================================
# ⚙️ 設定: ダブルエンジン (2連単 & 3連単 同時狙い)
# ==========================================
MODEL_FILE_3T = "boatrace_model.txt"    # 3連単用
MODEL_FILE_2T = "boatrace_model_2t.txt" # 2連単用

# ----------------------------------------------------
# 📊 戦略設定 (シミュレーション結果の完全移植)
# ----------------------------------------------------

# 【二連単】 回収率130% 厳選設定
STRATEGY_2T = {
    8:  4.0,  # 常滑
    10: 4.0,  # 三国
    16: 3.0,  # 児島
    21: 2.5,  # 芦屋
}

# 【三連単】 回収率124% 攻撃設定 (2022除外Sim結果)
STRATEGY_3T = {
    2:  2.0,  # 戸田
    3:  1.2,  # 江戸川
    5:  2.0,  # 多摩川
    6:  1.6,  # 浜名湖
    8:  1.8,  # 常滑 (2連単と重複！両方狙う)
    9:  1.4,  # 津
    10: 1.3,  # 三国 (重複！)
    11: 2.5,  # びわこ
    13: 1.6,  # 住之江
    14: 1.6,  # 尼崎
    16: 1.5,  # 児島 (重複！)
    19: 1.3,  # 下関
    20: 2.0,  # 若松
    22: 1.2,  # 福岡
    23: 1.5,  # 唐津
    24: 1.5,  # 大村
}

# 共通パラメータ
MIN_PROB_THRESHOLD = 0.0005     # 3連単に合わせて極限まで下げる
MAX_BETS_PER_RACE = 10          # 両方買う可能性があるので少し広げる
CALC_ODDS_CAP = 300.0           # 3連単に合わせて上限開放

# ==========================================
# 🧠 モデル管理
# ==========================================
MODELS = {'3t': None, '2t': None}

def load_models():
    if MODELS['2t'] is None and os.path.exists(MODEL_FILE_2T):
        print(f"📂 2連単モデル読込: {MODEL_FILE_2T}")
        MODELS['2t'] = lgb.Booster(model_file=MODEL_FILE_2T)
    
    if MODELS['3t'] is None:
        if os.path.exists(MODEL_FILE_3T):
            print(f"📂 3連単モデル読込: {MODEL_FILE_3T}")
            MODELS['3t'] = lgb.Booster(model_file=MODEL_FILE_3T)

def to_float(val):
    try: return float(val) if val else 0.0
    except: return 0.0

# ==========================================
# 🔮 予測 & 候補出し (両対応)
# ==========================================
def predict_race(raw):
    """
    戻り値: candidates (リスト)
    各候補に 'type': '2t' または '3t' が付与される
    """
    load_models()
    jcd = raw.get('jcd', 0)
    
    # この会場で有効な戦略があるかチェック
    use_2t = jcd in STRATEGY_2T
    use_3t = jcd in STRATEGY_3T
    
    if not use_2t and not use_3t:
        return [] # 戦略対象外

    # 特徴量作成
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
    
    if sum(ex_list) == 0: return []

    df_race = pd.DataFrame(rows)
    for col in ['wr', 'mo', 'ex', 'st']:
        mean = df_race[col].mean(); std = df_race[col].std()
        if std == 0: std = 1e-6
        df_race[f'{col}_z'] = (df_race[col] - mean) / std

    df_race['jcd'] = df_race['jcd'].astype('category')
    df_race['pid'] = df_race['pid'].astype('category')
    features = ['jcd', 'boat_no', 'pid', 'wind', 'wr', 'mo', 'ex', 'st', 'f', 'wr_z', 'mo_z', 'ex_z', 'st_z']
    
    candidates = []
    b = df_race['boat_no'].values

    # -------- 2連単 予測 --------
    if use_2t and MODELS['2t']:
        try:
            preds = MODELS['2t'].predict(df_race[features])
            p1, p2 = preds[:, 0], preds[:, 1]
            for i, j in permutations(range(6), 2):
                score = p1[i] * p2[j]
                if score >= 0.01: # 2連単は1%以上で足切り
                    candidates.append({
                        'combo': f"{b[i]}-{b[j]}",
                        'raw_prob': score,
                        'prob': round(score * 100, 1),
                        'type': '2t'
                    })
        except: pass

    # -------- 3連単 予測 --------
    if use_3t and MODELS['3t']:
        try:
            preds = MODELS['3t'].predict(df_race[features])
            p1, p2, p3 = preds[:, 0], preds[:, 1], preds[:, 2]
            for i, j, k in permutations(range(6), 3):
                score = p1[i] * p2[j] * p3[k]
                if score >= MIN_PROB_THRESHOLD: # 3連単は0.05%以上
                    candidates.append({
                        'combo': f"{b[i]}-{b[j]}-{b[k]}",
                        'raw_prob': score,
                        'prob': round(score * 100, 1),
                        'type': '3t'
                    })
        except: pass

    # 確率順にソートして返す
    candidates.sort(key=lambda x: x['raw_prob'], reverse=True)
    return candidates

# ==========================================
# 💰 EVフィルタ (2t/3t 混合対応)
# ==========================================
def filter_and_sort_bets(candidates, odds_2t_map, odds_3t_map, jcd):
    final_bets = []
    max_ev = 0.0
    thresh_info = 0.0

    # その会場の基準値を取得
    thresh_2t = STRATEGY_2T.get(jcd, 99.9)
    thresh_3t = STRATEGY_3T.get(jcd, 99.9)

    for bet in candidates:
        combo = bet['combo']
        prob = bet['raw_prob']
        b_type = bet['type']
        
        # タイプに応じたオッズと閾値を選択
        if b_type == '2t':
            real_odds = odds_2t_map.get(combo, 0.0)
            threshold = thresh_2t
        else:
            real_odds = odds_3t_map.get(combo, 0.0)
            threshold = thresh_3t

        if real_odds == 0: continue
        
        # キャップ適用 (3連単は300倍、2連単は100倍にしておく)
        cap = 300.0 if b_type == '3t' else 100.0
        calc_odds = min(real_odds, cap)
        
        ev = prob * calc_odds
        
        if ev > max_ev: 
            max_ev = ev
            thresh_info = threshold # ログ表示用
        
        if ev >= threshold:
            bet['odds'] = real_odds
            bet['ev'] = ev
            bet['reason'] = f"EV:{ev:.2f} (基準{threshold})"
            final_bets.append(bet)
            
    final_bets.sort(key=lambda x: x['ev'], reverse=True)
    return final_bets[:MAX_BETS_PER_RACE], max_ev, thresh_info

def attach_reason(results, raw, odds_map):
    for item in results:
        item['reason'] = f"【勝負】AI厳選 ({item['type'].upper()}) EV:{item['ev']:.2f}"
