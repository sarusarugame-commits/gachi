import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os
import re
import traceback
from openai import OpenAI # ★変更: OpenAIライブラリを使用

MODEL_FILE = 'ultimate_boat_model.pkl'
STRATEGY_FILE = 'ultimate_winning_strategies.csv'

# ==========================================
# ⚙️ 本番運用設定
# ==========================================
MIN_PROFIT = 1000   # 期待値1000円以上のみ通知
MIN_ROI = 110       # 回収率110%以上のみ通知

# Groq設定 (OpenAI互換エンドポイント)
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

client = None
if os.environ.get("GROQ_API_KEY"):
    client = OpenAI(
        api_key=os.environ.get("GROQ_API_KEY"),
        base_url=GROQ_BASE_URL # ★重要: ここでGroqに向ける
    )

def ask_groq_reason(row, combo, ptype):
    print(f"🤖 Groq API呼び出し(OpenAI Client): {combo} ({ptype})...", flush=True)
    
    if not client: 
        print("❌ Groq Error: APIキーが設定されていません", flush=True)
        return "AI解説: (APIキー設定なし)"
    
    try:
        def safe_get(key):
            try:
                val = row.get(key, 0)
                if isinstance(val, (list, np.ndarray)):
                    return val[0] if len(val) > 0 else 0
                return val
            except:
                return 0
            
        data_str = (
            f"1号艇:勝率{safe_get('wr1')}\n"
            f"2号艇:勝率{safe_get('wr2')}\n"
            f"3号艇:勝率{safe_get('wr3')}\n"
            f"4号艇:勝率{safe_get('wr4')}\n"
        )
        prompt = f"買い目「{combo}」({ptype})を推奨する理由を、競艇のプロとして100文字以内で断言せよ。\nデータ:\n{data_str}"
        
        # OpenAIライブラリの書き方はGroqと同じ
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a professional boat race analyst. Answer in Japanese."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150,
        )
        content = completion.choices[0].message.content
        print(f"🤖 Groq応答成功: {content[:20]}...", flush=True)
        return content

    except Exception as e:
        print(f"❌ Groq API Error: {e}", flush=True)
        return f"AI解説エラー: {str(e)}"

# 再帰的クリーニング
def unwrap_value(v):
    if isinstance(v, (list, tuple, np.ndarray)):
        if len(v) == 0: return 0.0
        return unwrap_value(v[0])
    if isinstance(v, str):
        try:
            return float(v.replace(',', '').replace('[','').replace(']','').strip())
        except:
            return 0.0
    try:
        return float(v)
    except:
        return 0.0

def predict_race(raw_data):
    recommendations = []
    
    # ---------------------------------------------------------
    # 0. 前処理
    # ---------------------------------------------------------
    clean_data = {}
    for k, v in raw_data.items():
        clean_data[k] = unwrap_value(v)
            
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
            return []

        df = pd.DataFrame([clean_data])
        
        for i in range(1, 7):
            if f'wr{i}' not in df.columns: df[f'wr{i}'] = 0.0
            if f'mo{i}' not in df.columns: df[f'mo{i}'] = 0.0
            if f'ex{i}' not in df.columns: df[f'ex{i}'] = 0.0
            if f'st{i}' not in df.columns: df[f'st{i}'] = 0.0

        df['wr_mean'] = df[[f'wr{i}' for i in range(1, 7)]].mean(axis=1)
        df['mo_mean'] = df[[f'mo{i}' for i in range(1, 7)]].mean(axis=1)
        df['ex_mean'] = df[[f'ex{i}' for i in range(1, 7)]].mean(axis=1)
        df['st_mean'] = df[[f'st{i}' for i in range(1, 7)]].mean(axis=1)

        for i in range(1, 7):
            df[f'wr{i}_rel'] = df[f'wr{i}'] - df['wr_mean']
            df[f'mo{i}_rel'] = df[f'mo{i}'] - df['mo_mean']
            df[f'ex{i}_rel'] = df['ex_mean'] - df[f'ex{i}'] 
            df[f'st{i}_rel'] = df['st_mean'] - df[f'st{i}'] 
        
        df_final = pd.DataFrame()
        for f in required_feats:
            if f in df.columns:
                df_final[f] = df[f]
            else:
                df_final[f] = 0.0
        
        X = df_final.values.astype(np.float32)
        
        try:
            def safe_predict_idx(model, input_x):
                try:
                    proba = model.predict_proba(input_x)
                    return np.argmax(proba, axis=1)[0]
                except:
                    pass
                
                pred = model.predict(input_x)
                if hasattr(pred, 'ndim') and pred.ndim == 2 and pred.shape[1] > 1:
                    return np.argmax(pred, axis=1)[0]
                
                val = pred[0]
                if hasattr(val, 'ndim') and val.ndim > 0:
                    if val.size == 1:
                        val = val.item()
                    else:
                        try:
                            return np.argmax(val)
                        except:
                            val = val[0]
                elif isinstance(val, (list, tuple)) and len(val) > 1:
                     val = val[0]
                
                return int(val) - 1

            p1_idx = safe_predict_idx(models['r1'], X)
            p2_idx = safe_predict_idx(models['r2'], X)
            p3_idx = safe_predict_idx(models['r3'], X)

        except Exception as inner_e:
            print(f"⚠️ Internal Predict Error: {inner_e}")
            return []

        p1, p2, p3 = p1_idx + 1, p2_idx + 1, p3_idx + 1
        
    except Exception as e:
        print(f"⚠️ AI Prediction Error: {e}", flush=True)
        return [] 

    # ---------------------------------------------------------
    # 2. 買い目作成とフィルタリング
    # ---------------------------------------------------------
    form_3t = f"{p1}-{p2}-{p3}"
    form_2t = f"{p1}-{p2}"
    
    strategies = None
    try:
        if os.path.exists(STRATEGY_FILE):
            strategies = pd.read_csv(STRATEGY_FILE)
    except: pass

    # ★ 3連単チェック
    if p1 != p2 and p1 != p3 and p2 != p3:
        profit, prob, roi = 0, 0, 0
        valid = False
        
        if strategies is not None:
            match = strategies[(strategies['券種'] == '3連単') & (strategies['買い目'] == form_3t)]
            if not match.empty:
                profit = int(match.iloc[0]['収支'])
                prob = match.iloc[0]['的中率']
                roi = match.iloc[0]['回収率']
                
                if profit >= MIN_PROFIT and roi >= MIN_ROI:
                    valid = True
                    print(f"✅ 採用: 3連単 {form_3t} (期待値:{profit}円/ROI:{roi}%)", flush=True)
                else:
                    print(f"🛑 却下: 3連単 {form_3t} (期待値:{profit}円/ROI:{roi}%) - 基準不足", flush=True)
            else:
                print(f"⚠️ データなし: 3連単 {form_3t}", flush=True)
        
        if valid:
            reason = ask_groq_reason(clean_data, form_3t, "3連単")
            recommendations.append({
                'type': '3連単',
                'combo': form_3t,
                'prob': prob,
                'profit': profit,
                'roi': roi,
                'reason': reason
            })

    # ★ 2連単チェック
    if p1 != p2:
        profit, prob, roi = 0, 0, 0
        valid = False
        
        if strategies is not None:
            match = strategies[(strategies['券種'] == '2連単') & (strategies['買い目'] == form_2t)]
            if not match.empty:
                profit = int(match.iloc[0]['収支'])
                prob = match.iloc[0]['的中率']
                roi = match.iloc[0]['回収率']
                
                if profit >= MIN_PROFIT and roi >= MIN_ROI:
                    valid = True
                    print(f"✅ 採用: 2連単 {form_2t} (期待値:{profit}円/ROI:{roi}%)", flush=True)
                else:
                    print(f"🛑 却下: 2連単 {form_2t} (期待値:{profit}円/ROI:{roi}%) - 基準不足", flush=True)
            else:
                print(f"⚠️ データなし: 2連単 {form_2t}", flush=True)
        
        if valid:
            reason = ask_groq_reason(clean_data, form_2t, "2連単")
            recommendations.append({
                'type': '2連単',
                'combo': form_2t,
                'prob': prob,
                'profit': profit,
                'roi': roi,
                'reason': reason
            })
            
    return recommendations
