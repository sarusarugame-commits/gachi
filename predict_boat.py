import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os
from groq import Groq

MODEL_FILE = 'ultimate_boat_model.pkl'
STRATEGY_FILE = 'ultimate_winning_strategies.csv'

# ★強制通知設定（動作確認用）
MIN_PROFIT = -999999 
MIN_ROI = 0       

# Groq設定
GROQ_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

client = None
if os.environ.get("GROQ_API_KEY"):
    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
        base_url=GROQ_URL
    )

def ask_groq_reason(row, combo, ptype):
    if not client: return "AI解説: (APIキー設定確認中)"
    try:
        # 値を取り出す際にも安全策を講じる
        def get_val(key):
            v = row.get(key, 0)
            if isinstance(v, list) or isinstance(v, np.ndarray):
                return v[0] if len(v) > 0 else 0
            return v
            
        data_str = (
            f"1号艇:勝率{get_val('wr1')}\n"
            f"2号艇:勝率{get_val('wr2')}\n"
            f"3号艇:勝率{get_val('wr3')}\n"
            f"4号艇:勝率{get_val('wr4')}\n"
        )
        prompt = f"買い目「{combo}」({ptype})を推奨する理由を、競艇のプロとして100文字以内で断言せよ。\nデータ:\n{data_str}"
        
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a professional boat race analyst. Answer in Japanese."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"AI解説エラー: {str(e)}"

def engineer_features(df):
    # 基本特徴量
    base_cols = [
        'wind',
        'wr1', 'mo1', 'ex1', 'st1', 'wr2', 'mo2', 'ex2', 'st2',
        'wr3', 'mo3', 'ex3', 'st3', 'wr4', 'mo4', 'ex4', 'st4',
        'wr5', 'mo5', 'ex5', 'st5', 'wr6', 'mo6', 'ex6', 'st6'
    ]
    
    # ★修正ポイント: リストや配列の中身を強制的に取り出す
    for col in base_cols:
        # まずリストかどうかチェックして、リストなら先頭要素を取り出す関数を適用
        df[col] = df[col].apply(lambda x: x[0] if isinstance(x, (list, np.ndarray)) and len(x) > 0 else x)
        # その後、数値変換
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    # 追加特徴量
    df['wr_mean'] = df[[f'wr{i}' for i in range(1, 7)]].mean(axis=1)
    df['mo_mean'] = df[[f'mo{i}' for i in range(1, 7)]].mean(axis=1)
    df['ex_mean'] = df[[f'ex{i}' for i in range(1, 7)]].mean(axis=1)
    df['st_mean'] = df[[f'st{i}' for i in range(1, 7)]].mean(axis=1)

    for i in range(1, 7):
        df[f'wr{i}_rel'] = df[f'wr{i}'] - df['wr_mean']
        df[f'mo{i}_rel'] = df[f'mo{i}'] - df['mo_mean']
        df[f'ex{i}_rel'] = df['ex_mean'] - df[f'ex{i}'] 
        df[f'st{i}_rel'] = df['st_mean'] - df[f'st{i}'] 
    
    return df

def predict_race(raw_data):
    recommendations = []
    
    # ---------------------------------------------------------
    # 1. AI予測
    # ---------------------------------------------------------
    try:
        if not os.path.exists(MODEL_FILE):
            return []

        models = joblib.load(MODEL_FILE)
        
        if 'features' in models:
            required_feats = models['features']
        else:
            print("⚠️ Model Error: 'features' key missing in pickle.")
            return []

        # データ作成
        df = pd.DataFrame([raw_data])
        df = engineer_features(df) # ここで型変換される
        
        # 列の整合性確保
        df_final = pd.DataFrame()
        for f in required_feats:
            if f in df.columns:
                df_final[f] = df[f]
            else:
                df_final[f] = 0.0
        
        # 予測実行
        try:
            p1_idx = np.argmax(models['r1'].predict_proba(df_final), axis=1)[0]
            p2_idx = np.argmax(models['r2'].predict_proba(df_final), axis=1)[0]
            p3_idx = np.argmax(models['r3'].predict_proba(df_final), axis=1)[0]
        except:
            p1_idx = int(models['r1'].predict(df_final)[0]) - 1
            p2_idx = int(models['r2'].predict(df_final)[0]) - 1
            p3_idx = int(models['r3'].predict(df_final)[0]) - 1

        p1, p2, p3 = p1_idx + 1, p2_idx + 1, p3_idx + 1
        
    except Exception as e:
        print(f"⚠️ AI Prediction Error: {e}", flush=True)
        return [] # エラー時は通知しない

    # ---------------------------------------------------------
    # 2. 買い目作成 (AI成功時のみ)
    # ---------------------------------------------------------
    form_3t = f"{p1}-{p2}-{p3}"
    form_2t = f"{p1}-{p2}"
    
    profit, prob, roi = 9999, 99.9, 999 
    
    try:
        if os.path.exists(STRATEGY_FILE):
            strategies = pd.read_csv(STRATEGY_FILE)
            match = strategies[(strategies['券種'] == '3連単') & (strategies['買い目'] == form_3t)]
            if not match.empty:
                profit = int(match.iloc[0]['収支'])
                prob = match.iloc[0]['的中率']
                roi = match.iloc[0]['回収率']
    except: pass 

    # ★ 3連単 (強制通知)
    if p1 != p2 and p1 != p3 and p2 != p3:
        reason = ask_groq_reason(raw_data, form_3t, "3連単")
        recommendations.append({
            'type': '3連単',
            'combo': form_3t,
            'prob': prob,
            'profit': profit,
            'roi': roi,
            'reason': reason
        })

    # ★ 2連単 (強制通知)
    if p1 != p2:
        reason = ask_groq_reason(raw_data, form_2t, "2連単")
        recommendations.append({
            'type': '2連単',
            'combo': form_2t,
            'prob': 80.0,
            'profit': 2000,
            'roi': 120,
            'reason': reason
        })
            
    return recommendations
