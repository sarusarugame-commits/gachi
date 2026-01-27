import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os

# ==========================================
# ⚙️ 設定エリア
# ==========================================
MODEL_FILE = 'ultimate_boat_model.pkl'
STRATEGY_FILE = 'ultimate_winning_strategies.csv'

# 厳選フィルタ（上振れ排除設定）
MIN_PROFIT = 1000  # 収支1000円以上のみ対象
MIN_ROI = 110      # 回収率110%以上のみ対象

# モデル特徴量（学習時と完全に一致させること）
BASE_FEATURES = [
    'wind',
    'wr1', 'mo1', 'ex1', 'st1',
    'wr2', 'mo2', 'ex2', 'st2',
    'wr3', 'mo3', 'ex3', 'st3',
    'wr4', 'mo4', 'ex4', 'st4',
    'wr5', 'mo5', 'ex5', 'st5',
    'wr6', 'mo6', 'ex6', 'st6'
]

def engineer_features(df):
    """AIモデル用の特徴量（相対評価など）を作成"""
    # 平均値
    df['wr_mean'] = df[[f'wr{i}' for i in range(1, 7)]].mean(axis=1)
    df['mo_mean'] = df[[f'mo{i}' for i in range(1, 7)]].mean(axis=1)
    df['ex_mean'] = df[[f'ex{i}' for i in range(1, 7)]].mean(axis=1)
    df['st_mean'] = df[[f'st{i}' for i in range(1, 7)]].mean(axis=1)

    new_feats = []
    for i in range(1, 7):
        df[f'wr{i}_rel'] = df[f'wr{i}'] - df['wr_mean']
        df[f'mo{i}_rel'] = df[f'mo{i}'] - df['mo_mean']
        df[f'ex{i}_rel'] = df['ex_mean'] - df[f'ex{i}'] # タイムは小さい方が良い
        df[f'st{i}_rel'] = df['st_mean'] - df[f'st{i}'] # STも小さい方が良い
        new_feats.extend([f'wr{i}_rel', f'mo{i}_rel', f'ex{i}_rel', f'st{i}_rel'])
    
    return df[BASE_FEATURES + new_feats]

def predict_race(raw_data):
    """
    raw_data: scraper.scrape_race_dataで取得した辞書データ
    戻り値: 推奨買い目のリスト（なければ空リスト）
    """
    if not os.path.exists(MODEL_FILE) or not os.path.exists(STRATEGY_FILE):
        print("Error: Model or Strategy file not found.")
        return []

    try:
        # データフレーム化 & 特徴量生成
        df = pd.DataFrame([raw_data])
        df = engineer_features(df)
        
        # モデルロード
        models = joblib.load(MODEL_FILE)
        
        # 予測 (各モデルで最も確率の高い艇を選ぶ)
        p1 = np.argmax(models['r1'].predict(df), axis=1)[0]
        p2 = np.argmax(models['r2'].predict(df), axis=1)[0]
        p3 = np.argmax(models['r3'].predict(df), axis=1)[0]
        
        form_3t = f"{p1}-{p2}-{p3}"
        form_2t = f"{p1}-{p2}"
        best_boat = p1 # 本命艇

        # 戦略リスト読み込み & フィルタリング
        strategies = pd.read_csv(STRATEGY_FILE)
        valid_strategies = strategies[
            (strategies['回収率'] >= MIN_ROI) & 
            (strategies['収支'] >= MIN_PROFIT)
        ]
        
        valid_3t = set(valid_strategies[valid_strategies['券種']=='3連単']['買い目'])
        valid_2t = set(valid_strategies[valid_strategies['券種']=='2連単']['買い目'])
        
        recommendations = []

        # 3連単チェック
        if form_3t in valid_3t:
            if p1!=p2 and p1!=p3 and p2!=p3:
                row = valid_strategies[(valid_strategies['券種']=='3連単') & (valid_strategies['買い目']==form_3t)].iloc[0]
                recommendations.append({
                    'type': '3連単',
                    'combo': form_3t,
                    'prob': row['的中率']/100, # csvの的中率は%表記なので
                    'roi': row['回収率'],
                    'profit': row['収支'],
                    'best_boat': best_boat
                })

        # 2連単チェック
        if form_2t in valid_2t:
            if p1!=p2:
                row = valid_strategies[(valid_strategies['券種']=='2連単') & (valid_strategies['買い目']==form_2t)].iloc[0]
                recommendations.append({
                    'type': '2連単',
                    'combo': form_2t,
                    'prob': row['的中率']/100,
                    'roi': row['回収率'],
                    'profit': row['収支'],
                    'best_boat': best_boat
                })
                
        return recommendations

    except Exception as e:
        print(f"Predict Error: {e}")
        return []
