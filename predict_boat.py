import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os
from groq import Groq

MODEL_FILE = 'ultimate_boat_model.pkl'
STRATEGY_FILE = 'ultimate_winning_strategies.csv'

# ★強制通知モード（条件無視）
MIN_PROFIT = -999999 
MIN_ROI = 0       

BASE_FEATURES = [
    'wind',
    'wr1', 'mo1', 'ex1', 'st1',
    'wr2', 'mo2', 'ex2', 'st2',
    'wr3', 'mo3', 'ex3', 'st3',
    'wr4', 'mo4', 'ex4', 'st4',
    'wr5', 'mo5', 'ex5', 'st5',
    'wr6', 'mo6', 'ex6', 'st6'
]

# ★ご指定の設定値を定数化
GROQ_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

client = None
if os.environ.get("GROQ_API_KEY"):
    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
        base_url=GROQ_URL  # ★ご指定のURLを適用
    )

def ask_groq_reason(row, combo, ptype):
    if not client: return "AI解説: (APIキー設定確認中)"
    try:
        data_str = (
            f"風速:{row.get('wind',0)}m\n"
            f"1号艇:勝率{row.get('wr1',0)}\n"
            f"2号艇:勝率{row.get('wr2',0)}\n"
            f"3号艇:勝率{row.get('wr3',0)}\n"
            f"4号艇:勝率{row.get('wr4',0)}\n"
        )
        prompt = f"買い目「{combo}」({ptype})が激熱な理由を、レース展開を妄想して100文字以内で断言しろ。\nデータ:\n{data_str}"
        
        completion = client.chat.completions.create(
            model=GROQ_MODEL, # ★ご指定のモデルIDを適用
            messages=[
                {"role": "system", "content": "You are a professional boat race analyst. Answer in Japanese."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150,
        )
        return completion.choices[0].message.content
    except Exception as e:
        # 万が一モデル名などでエラーが出ても、ログに残して通知は止めない
        return f"AI解説エラー: {str(e)}"

def engineer_features(df):
    cols_to_convert = BASE_FEATURES
    for col in cols_to_convert:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    df['wr_mean'] = df[[f'wr{i}' for i in range(1, 7)]].mean(axis=1)
    df['mo_mean'] = df[[f'mo{i}' for i in range(1, 7)]].mean(axis=1)
    df['ex_mean'] = df[[f'ex{i}' for i in range(1, 7)]].mean(axis=1)
    df['st_mean'] = df[[f'st{i}' for i in range(1, 7)]].mean(axis=1)

    new_feats = []
    for i in range(1, 7):
        df[f'wr{i}_rel'] = df[f'wr{i}'] - df['wr_mean']
        df[f'mo{i}_rel'] = df[f'mo{i}'] - df['mo_mean']
        df[f'ex{i}_rel'] = df['ex_mean'] - df[f'ex{i}'] 
        df[f'st{i}_rel'] = df['st_mean'] - df[f'st{i}'] 
        new_feats.extend([f'wr{i}_rel', f'mo{i}_rel', f'ex{i}_rel', f'st{i}_rel'])
    
    return df[BASE_FEATURES + new_feats]

def predict_race(raw_data):
    if not os.path.exists(MODEL_FILE):
        return []

    try:
        df = pd.DataFrame([raw_data])
        df_features = engineer_features(df)
        models = joblib.load(MODEL_FILE)
        
        try:
            p1_idx = np.argmax(models['r1'].predict_proba(df_features), axis=1)[0]
            p2_idx = np.argmax(models['r2'].predict_proba(df_features), axis=1)[0]
            p3_idx = np.argmax(models['r3'].predict_proba(df_features), axis=1)[0]
        except:
            p1_idx = int(models['r1'].predict(df_features)[0]) - 1
            p2_idx = int(models['r2'].predict(df_features)[0]) - 1
            p3_idx = int(models['r3'].predict(df_features)[0]) - 1

        p1, p2, p3 = p1_idx + 1, p2_idx + 1, p3_idx + 1
        
        form_3t = f"{p1}-{p2}-{p3}"
        form_2t = f"{p1}-{p2}"
        
        strategies = None
        if os.path.exists(STRATEGY_FILE):
            strategies = pd.read_csv(STRATEGY_FILE)

        recommendations = []

        # --- 3連単 (強制通知) ---
        if p1 != p2 and p1 != p3 and p2 != p3:
            profit = 9999 
            prob = 99.9   
            roi = 999     
            
            if strategies is not None:
                match = strategies[(strategies['券種'] == '3連単') & (strategies['買い目'] == form_3t)]
                if not match.empty:
                    profit = int(match.iloc[0]['収支'])
                    prob = match.iloc[0]['的中率']
                    roi = match.iloc[0]['回収率']
            
            reason = ask_groq_reason(raw_data, form_3t, "3連単")
            
            recommendations.append({
                'type': '3連単',
                'combo': form_3t,
                'prob': prob,
                'profit': profit,
                'roi': roi,
                'reason': reason
            })

        # --- 2連単 (強制通知) ---
        if p1 != p2:
            reason = ask_groq_reason(raw_data, form_2t, "2連単")
            recommendations.append({
                'type': '2連単',
                'combo': form_2t,
                'prob': 88.8, 
                'profit': 5000,
                'roi': 150,
                'reason': reason
            })
                
        return recommendations

    except Exception as e:
        return []
