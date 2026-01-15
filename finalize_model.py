import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import os

# ==========================================
# ⚙️ 設定エリア
# ==========================================
CSV_PATH = r"C:\Users\TAKUMA\競艇に勝つ\競艇データ\FINAL_FULL_DATA_2025_FIXED.csv"

# 1. 特徴量生成（最強のシナジーモデルを継承）
def engineer_features(df):
    for i in range(1, 7):
        df[f'power_idx_{i}'] = df[f'wr{i}'] * (1.0 / (df[f'st{i}'] + 0.01))
    for i in range(1, 6):
        df[f'st_gap_{i}_{i+1}'] = df[f'st{i+1}'] - df[f'st{i}']
        df[f'wr_gap_{i}_{i+1}'] = df[f'wr{i}'] - df[f'wr{i+1}']
    avg_wr = df[[f'wr{i}' for i in range(1, 7)]].mean(axis=1)
    df['wr_1_vs_avg'] = df['wr1'] / (avg_wr + 0.001)
    df['jcd'] = df['jcd'].astype('category')
    return df

# 2. データ準備
df = pd.read_csv(CSV_PATH).dropna(subset=['rank1', 'rank2'])
df = engineer_features(df)

features = ['jcd', 'rno', 'wind', 'wr_1_vs_avg']
for i in range(1, 7):
    features.extend([f'wr{i}', f'st{i}', f'ex{i}', f'power_idx_{i}'])
for i in range(1, 6):
    features.extend([f'st_gap_{i}_{i+1}', f'wr_gap_{i}_{i+1}'])

# 正解ラベル（二連単）
combinations = [f"{f}-{s}" for f in range(1, 7) for s in range(1, 7) if f != s]
combo_to_id = {c: i for i, c in enumerate(combinations)}
df['target'] = (df['rank1'].astype(int).astype(str) + "-" + df['rank2'].astype(int).astype(str)).map(combo_to_id)
df = df.dropna(subset=['target'])

# 3. 最終学習（全データを使用）
print("🧠 最終モデルを構築中...")
lgb_train = lgb.Dataset(df[features], label=df['target'])
params = {
    'objective': 'multiclass', 'num_class': 30, 'metric': 'multi_logloss',
    'num_leaves': 127, 'learning_rate': 0.01, 'verbose': -1, 'seed': 42
}
model = lgb.train(params, lgb_train, num_boost_round=1000)

# 4. モデルの保存（これで実戦でいつでも呼び出せます）
model.save_model('boat_model_nirentan.txt')
print("✅ モデルを 'boat_model_nirentan.txt' として保存しました。")

# 5. 特徴量重要度の可視化（AIが何を重視しているか？）
importances = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)

print("\n📍 AIが重視しているデータ TOP10:")
print(importances.head(10))

# 簡易グラフ表示
plt.figure(figsize=(10, 6))
plt.barh(importances['feature'].head(15), importances['importance'].head(15))
plt.gca().invert_yaxis()
plt.title("Feature Importance (AI's Eye)")
plt.show()