import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import zipfile  # 追加
from itertools import permutations

# ==========================================
# ⚙️ 設定・戦略
# ==========================================
MODEL_FILE = "boat_race_model_3t.txt"

# グローバル変数でモデルを保持
AI_MODEL = None

# 【会場別】最適戦略ポートフォリオ
STRATEGY = {
    1:  {'th': 0.065, 'k': 1},  # 桐生
    2:  {'th': 0.050, 'k': 5},  # 戸田
    3:  {'th': 0.060, 'k': 8},  # 江戸川
    4:  {'th': 0.050, 'k': 5},  # 平和島
    5:  {'th': 0.040, 'k': 1},  # 多摩川
    7:  {'th': 0.065, 'k': 1},  # 蒲郡
    8:  {'th': 0.070, 'k': 5},  # 常滑
    9:  {'th': 0.055, 'k': 1},  # 津
    10: {'th': 0.060, 'k': 8},  # 三国
    11: {'th': 0.045, 'k': 1},  # びわこ
    12: {'th': 0.060, 'k': 1},  # 住之江
    13: {'th': 0.040, 'k': 1},  # 尼崎
    15: {'th': 0.065, 'k': 1},  # 丸亀
    16: {'th': 0.055, 'k': 1},  # 児島
    18: {'th': 0.070, 'k': 1},  # 徳山
    19: {'th': 0.065, 'k': 1},  # 下関
    20: {'th': 0.070, 'k': 8},  # 若松
    21: {'th': 0.060, 'k': 1},  # 芦屋
    22: {'th': 0.055, 'k': 1},  # 福岡
}

def load_model():
    """モデルをロード（Zip対応版）"""
    global AI_MODEL
    if AI_MODEL is None:
        # 1. txtがそのままあれば読む
        if os.path.exists(MODEL_FILE):
            AI_MODEL = lgb.Booster(model_file=MODEL_FILE)
        
        # 2. txtがないけどzipがあるなら、解凍してから読む
        elif os.path.exists(MODEL_FILE.replace(".txt", ".zip")):
            zip_path = MODEL_FILE.replace(".txt", ".zip")
            print(f"📦 モデル圧縮ファイルを発見: {zip_path}")
            try:
                with zipfile.ZipFile(zip_path, 'r') as z:
                    z.extractall(".") # カレントディレクトリに解凍
                print("✅ 解凍成功！モデルをロードします。")
                AI_MODEL = lgb.Booster(model_file=MODEL_FILE)
            except Exception as e:
                print(f"❌ モデル解凍エラー: {e}")
                return None
        else:
            print(f"❌ モデルファイルが見つかりません: {MODEL_FILE}")
            return None
    return AI_MODEL

def predict_race(raw):
    """
    main.py から渡された raw データ (dict) を使って予測する
    """
    model = load_model()
    if model is None: return []

    jcd = raw.get('jcd', 0)
    wind = raw.get('wind', 0.0)
    
    if jcd not in STRATEGY: return []

    # 展示タイムなしはスキップ
    has_ex = sum([raw.get(f'ex{i}', 0) for i in range(1, 7)]) > 0
    if not has_ex: return []

    rows = []
    for i in range(1, 7):
        s = str(i)
        row = {
            'jcd': jcd,
            'wind': wind,
            'boat_no': i,
            'pid': raw.get(f'pid{s}', 0),
            'wr': raw.get(f'wr{s}', 0.0),
            'mo': raw.get(f'mo{s}', 0.0),
            'ex': raw.get(f'ex{s}', 0.0),
            'st': raw.get(f'st{s}', 0.20),
            'f': raw.get(f'f{s}', 0),
        }
        rows.append(row)
    
    df_race = pd.DataFrame(rows)

    # 前処理
    for col in ['wr', 'mo', 'ex', 'st']:
        mean = df_race[col].mean()
        std = df_race[col].std()
        if std == 0: std = 1e-6
        df_race[f'{col}_z'] = (df_race[col] - mean) / std

    df_race['jcd'] = df_race['jcd'].astype('category')
    df_race['pid'] = df_race['pid'].astype('category')
    
    features = [
        'jcd', 'boat_no', 'wind', 'pid',
        'wr', 'mo', 'ex', 'st', 'f',
        'wr_z', 'mo_z', 'ex_z', 'st_z'
    ]

    try:
        preds = model.predict(df_race[features])
        if preds.shape[1] < 3: return [] 
        p1_arr, p2_arr, p3_arr = preds[:, 0], preds[:, 1], preds[:, 2]
    except Exception:
        return []

    # 3連単全通り
    b = df_race['boat_no'].values
    combos = []
    for i, j, k in permutations(range(6), 3):
        score = p1_arr[i] * p2_arr[j] * p3_arr[k]
        combos.append({
            'combo': f"{b[i]}-{b[j]}-{b[k]}",
            'score': score
        })
    
    combos.sort(key=lambda x: x['score'], reverse=True)
    
    strat = STRATEGY[jcd]
    if combos[0]['score'] >= strat['th']:
        return [{
            'combo': item['combo'],
            'type': f"自信度{item['score']:.4f}",
            'profit': 0,
            'prob': int(item['score'] * 100),
            'roi': 0,
            'reason': f"戦略適合(基準{strat['th']})"
        } for item in combos[:strat['k']]]

    return []
