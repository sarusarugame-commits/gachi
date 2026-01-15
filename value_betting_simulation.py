import pandas as pd
import numpy as np
import lightgbm as lgb
import time
import os

# ==========================================
# ⚙️ 設定エリア
# ==========================================
CSV_PATH = r"C:\Users\TAKUMA\競艇に勝つ\競艇データ\FINAL_FULL_DATA_2025_FIXED.csv"
EV_THRESHOLD = 1.2  # 期待値1.2以上を「買い」と判定

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# 1. 特徴量生成（最強モデルと同じものを使用）
def engineer_features(df):
    for i in range(1, 6):
        df[f'st_gap_{i}_{i+1}'] = df[f'st{i+1}'] - df[f'st{i}']
        df[f'wr_gap_{i}_{i+1}'] = df[f'wr{i}'] - df[f'wr{i+1}']
    avg_wr = df[[f'wr{i}' for i in range(1, 7)]].mean(axis=1)
    df['wr_1_vs_avg'] = df['wr1'] / (avg_wr + 0.001)
    df['jcd'] = df['jcd'].astype('category')
    return df

# 2. データ準備
log("📂 データを読み込んでいます...")
df = pd.read_csv(CSV_PATH).dropna(subset=['rank1', 'rank2', 'tansho', 'nirentan'])
df = engineer_features(df)

features = ['jcd', 'rno', 'wind', 'wr_1_vs_avg']
for i in range(1, 7):
    features.extend([f'wr{i}', f'st{i}', f'ex{i}'])
for i in range(1, 6):
    features.extend([f'st_gap_{i}_{i+1}', f'wr_gap_{i}_{i+1}'])

# 正解ラベル
df['target_tan'] = df['rank1'].astype(int) - 1
combinations = [f"{f}-{s}" for f in range(1, 7) for s in range(1, 7) if f != s]
combo_to_id = {c: i for i, c in enumerate(combinations)}
df['target_niren'] = (df['rank1'].astype(int).astype(str) + "-" + df['rank2'].astype(int).astype(str)).map(combo_to_id)
df = df.dropna(subset=['target_niren'])

# 分割
split_idx = int(len(df) * 0.8)
train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

# 3. 学習（検証用にサクッと学習させます）
log("🧠 AIの学習を実行中...")
model_tan = lgb.train({'objective':'multiclass','num_class':6,'verbose':-1}, 
                      lgb.Dataset(train_df[features], label=train_df['target_tan']), num_boost_round=100)
model_niren = lgb.train({'objective':'multiclass','num_class':30,'verbose':-1}, 
                        lgb.Dataset(train_df[features], label=train_df['target_niren']), num_boost_round=100)

# 4. 期待値シミュレーション
log("📊 期待値(EV)に基づいたシミュレーションを開始...")

def run_ev_analysis(model, test_data, payout_col, target_col, name, prob_th):
    probs = model.predict(test_data[features])
    conf = np.max(probs, axis=1)
    pred_class = np.argmax(probs, axis=1)
    
    # 確定オッズ（払戻金/100）
    odds = test_data[payout_col] / 100.0
    ev = conf * odds  # 🌟 期待値計算
    
    results = pd.DataFrame({
        'Prob': conf,
        'Odds': odds,
        'EV': ev,
        'Hit': pred_class == test_data[target_col].values,
        'Payout': test_data[payout_col]
    })
    
    # 手法A: 自信度だけで選別
    df_conf = results[results['Prob'] >= prob_th]
    # 手法B: 期待値(EV)で選別
    df_ev = results[(results['Prob'] >= (prob_th * 0.7)) & (results['EV'] >= EV_THRESHOLD)]
    
    print(f"\n--- 【{name}】 シミュレーション結果 ---")
    for label, d in [("自信度のみ", df_conf), ("期待値重視", df_ev)]:
        acc = d['Hit'].mean() * 100
        rec = (d['Hit'] * d['Payout']).sum() / (len(d) * 100) * 100
        print(f"[{label}] 的中率: {acc:5.2f}% | 回収率: {rec:6.2f}% | 購入数: {len(d):5d}R | 平均オッズ: {d['Odds'].mean():4.2f}倍")

run_ev_analysis(model_tan, test_df, 'tansho', 'target_tan', "単勝", 0.7)
run_ev_analysis(model_niren, test_df, 'nirentan', 'target_niren', "二連単", 0.3)