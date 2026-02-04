import pandas as pd
import numpy as np
import lightgbm as lgb
import os
from itertools import permutations

# ==========================================
# ⚙️ 設定: ダブルモデル & 厳選使い分け
# ==========================================
MODEL_FILE_3T = "boatrace_model.txt"    # 3連単用
MODEL_FILE_2T = "boatrace_model_2t.txt" # 2連単用 (新設)

# 【戦略】シミュレーション結果に基づく厳選設定
# キー: JCD, 値: {'mode': '2t' or '3t', 'ev_thresh': float}
# ここにない会場は見送り(Skip)
STRATEGY_MAP = {
    8:  {'mode': '2t', 'thresh': 4.0},  # 常滑
    10: {'mode': '2t', 'thresh': 4.0},  # 三国
    16: {'mode': '2t', 'thresh': 3.0},  # 蒲郡
    21: {'mode': '2t', 'thresh': 2.5},  # 芦屋 (エース)
}

# 共通フィルタ
MIN_PROB_THRESHOLD = 0.01
MAX_BETS_PER_RACE = 8
CALC_ODDS_CAP = 100.0

# ==========================================
# 🧠 モデル管理
# ==========================================
MODELS = {'3t': None, '2t': None}

def load_models():
    # 2連単モデル
    if MODELS['2t'] is None and os.path.exists(MODEL_FILE_2T):
        print(f"📂 2連単モデル読込: {MODEL_FILE_2T}")
        MODELS['2t'] = lgb.Booster(model_file=MODEL_FILE_2T)
    
    # 3連単モデル (今回は使わない設定だが、拡張用に残す)
    if MODELS['3t'] is None:
        if os.path.exists(MODEL_FILE_3T):
            print(f"📂 3連単モデル読込: {MODEL_FILE_3T}")
            MODELS['3t'] = lgb.Booster(model_file=MODEL_FILE_3T)

def to_float(val):
    try: return float(val) if val else 0.0
    except: return 0.0

# ==========================================
# 🔮 予測 & 候補出し
# ==========================================
def predict_race(raw):
    """
    戻り値: (候補リスト, モード('2t'/'3t'), 最大自信度)
    """
    load_models()
    jcd = raw.get('jcd', 0)
    
    # 戦略チェック: 買うべき会場か？
    strategy = STRATEGY_MAP.get(jcd)
    if not strategy:
        return [], None, 0.0 # 見送り
    
    mode = strategy['mode']
    model = MODELS.get(mode)
    
    if not model:
        # モデルファイルがない場合などはスキップ
        return [], None, 0.0

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
    
    if sum(ex_list) == 0: return [], None, 0.0

    df_race = pd.DataFrame(rows)
    for col in ['wr', 'mo', 'ex', 'st']:
        mean = df_race[col].mean(); std = df_race[col].std()
        if std == 0: std = 1e-6
        df_race[f'{col}_z'] = (df_race[col] - mean) / std

    df_race['jcd'] = df_race['jcd'].astype('category')
    df_race['pid'] = df_race['pid'].astype('category')
    features = ['jcd', 'boat_no', 'pid', 'wind', 'wr', 'mo', 'ex', 'st', 'f', 'wr_z', 'mo_z', 'ex_z', 'st_z']
    
    # 予測実行
    try:
        preds = model.predict(df_race[features])
        # p1(1着率), p2(2着率)
        p1 = preds[:, 0]
        p2 = preds[:, 1]
        # 3連単モデルの場合は p3 もあるが、今回は2連単メインなので無視or活用
        if mode == '3t': p3 = preds[:, 2]
    except: return [], None, 0.0

    max_prob = max(p1)
    candidates = []
    b = df_race['boat_no'].values

    # ★ 2連単モードの生成
    if mode == '2t':
        for i, j in permutations(range(6), 2):
            score = p1[i] * p2[j]
            if score >= MIN_PROB_THRESHOLD:
                candidates.append({
                    'combo': f"{b[i]}-{b[j]}",
                    'raw_prob': score,
                    'prob': round(score * 100, 1)
                })
                
    # ★ 3連単モードの生成 (もし使うなら)
    elif mode == '3t':
        for i, j, k in permutations(range(6), 3):
            score = p1[i] * p2[j] * p3[k]
            if score >= MIN_PROB_THRESHOLD:
                candidates.append({
                    'combo': f"{b[i]}-{b[j]}-{b[k]}",
                    'raw_prob': score,
                    'prob': round(score * 100, 1)
                })

    candidates.sort(key=lambda x: x['raw_prob'], reverse=True)
    return candidates[:50], mode, max_prob

# ==========================================
# 💰 EVフィルタ
# ==========================================
def filter_and_sort_bets(candidates, odds_map, jcd, mode):
    strategy = STRATEGY_MAP.get(jcd)
    threshold = strategy['thresh'] if strategy else 99.9
    
    final_bets = []
    max_ev = 0.0

    for bet in candidates:
        combo = bet['combo']
        prob = bet['raw_prob']
        
        real_odds = odds_map.get(combo, 0.0)
        if real_odds == 0: continue
        
        calc_odds = min(real_odds, CALC_ODDS_CAP)
        ev = prob * calc_odds
        
        if ev > max_ev: max_ev = ev
        
        if ev >= threshold:
            bet['odds'] = real_odds
            bet['ev'] = ev
            bet['type'] = mode # 2t or 3t
            bet['reason'] = f"EV:{ev:.2f} (基準{threshold})"
            final_bets.append(bet)
            
    final_bets.sort(key=lambda x: x['ev'], reverse=True)
    return final_bets[:MAX_BETS_PER_RACE], max_ev, threshold

def attach_reason(results, raw, odds_map):
    for item in results:
        item['reason'] = f"【勝負】AI厳選 ({item['type'].upper()}) EV:{item['ev']:.2f}"
