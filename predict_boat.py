import pandas as pd
import numpy as np
import lightgbm as lgb
import time
import os

# ==========================================
# ⚙️ 設定エリア
# ==========================================
CSV_PATH = r"C:\Users\TAKUMA\競艇に勝つ\競艇データ\FINAL_FULL_DATA_2025_FIXED.csv"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ==========================================
# 1. 特徴量エンジニアリング（シナジー・モデル）
# ==========================================
def engineer_synergy_features(df):
    log("🛠️ シナジー特徴量を生成中（勝率×スタートの相関など）...")
    
    # 選手の実力とスタートの掛け合わせ（最強の指標）
    for i in range(1, 7):
        # 勝率が高く、かつSTが早い（数値が小さい）ほど高い値になる指標
        df[f'power_idx_{i}'] = df[f'wr{i}'] * (1.0 / (df[f'st{i}'] + 0.01))
        
    # 1号艇と他艇の圧倒的格差
    df['top_power_gap'] = df['power_idx_1'] / (df[[f'power_idx_{i}' for i in range(2, 7)]].max(axis=1) + 0.001)
    
    # 会場ごとの平均的な「荒れ度」
    venue_hit_rate = df.groupby('jcd')['res1'].transform('mean')
    df['venue_stability'] = venue_hit_rate

    # 展示の相対評価（1号艇がどれだけ抜けているか）
    ex_mean = df[[f'ex{i}' for i in range(1, 7)]].mean(axis=1)
    df['ex_1_diff'] = ex_mean - df['ex1']

    df['jcd'] = df['jcd'].astype('category')
    return df

# データ準備
df = pd.read_csv(CSV_PATH).dropna(subset=['rank1', 'rank2', 'tansho', 'nirentan'])
df = engineer_synergy_features(df)

features = ['jcd', 'rno', 'wind', 'venue_stability', 'top_power_gap', 'ex_1_diff']
for i in range(1, 7):
    features.extend([f'wr{i}', f'st{i}', f'ex{i}', f'power_idx_{i}'])

# 正解ラベル
df['target_tan'] = df['rank1'].astype(int) - 1
combinations = [f"{f}-{s}" for f in range(1, 7) for s in range(1, 7) if f != s]
combo_to_id = {c: i for i, c in enumerate(combinations)}
df['target_niren'] = (df['rank1'].astype(int).astype(str) + "-" + df['rank2'].astype(int).astype(str)).map(combo_to_id)
df = df.dropna(subset=['target_niren'])

# 分割
split_idx = int(len(df) * 0.8)
train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

# ==========================================
# 2. 超・深層学習（限界までパラメータを追い込む）
# ==========================================
def train_limit_model(y_col, num_class):
    log(f"🧠 {y_col} の限界学習（最強パラメータ）を実行中...")
    lgb_train = lgb.Dataset(train_df[features], label=train_df[y_col])
    lgb_eval = lgb.Dataset(test_df[features], label=test_df[y_col], reference=lgb_train)
    
    params = {
        'objective': 'multiclass',
        'num_class': num_class,
        'metric': 'multi_logloss',
        'num_leaves': 511,         # 最大限の複雑さを許容
        'learning_rate': 0.002,    # 極限まで慎重に学習
        'feature_fraction': 0.6,
        'bagging_fraction': 0.6,
        'bagging_freq': 1,
        'min_data_in_leaf': 10,
        'lambda_l1': 1.0,          # 厳しいペナルティでノイズを排除
        'lambda_l2': 1.0,
        'verbose': -1,
        'seed': 42
    }
    
    return lgb.train(
        params, lgb_train, 
        num_boost_round=10000,     # 非常に長い学習
        valid_sets=[lgb_train, lgb_eval],
        callbacks=[lgb.early_stopping(stopping_rounds=300)]
    )

model_tan = train_limit_model('target_tan', 6)
model_niren = train_limit_model('target_niren', 30)

# ==========================================
# 3. 究極の限界分析
# ==========================================
def analyze_ultimate(model, name, is_niren=False):
    probs = model.predict(test_df[features])
    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)
    y_test = test_df['target_niren' if is_niren else 'target_tan'].values
    
    print(f"\n👑 【{name}】 究極限界分析結果")
    print("自信度 | 的中率 | レース数 | 回収率")
    print("-----------------------------------------")
    
    # さらに高い自信度をチェック
    thresholds = [0.85, 0.9, 0.92, 0.95] if not is_niren else [0.35, 0.4, 0.45, 0.5]
    
    for th in thresholds:
        mask = confs >= th
        if mask.sum() == 0: continue
        
        acc = (preds[mask] == y_test[mask]).mean() * 100
        payouts = test_df.iloc[mask]['nirentan' if is_niren else 'tansho']
        rec = (payouts[preds[mask] == y_test[mask]].sum() / (mask.sum() * 100)) * 100
        print(f"{th*100:2.0f}%  | {acc:6.2f}% | {mask.sum():5d}R | {rec:6.2f}%")

analyze_ultimate(model_tan, "単勝")
analyze_ultimate(model_niren, "二連単", True)