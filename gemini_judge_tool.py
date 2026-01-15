import pandas as pd
import numpy as np
import lightgbm as lgb
import google.generativeai as genai

# ==========================================
# ⚙️ 設定エリア
# ==========================================
# Gemini APIキーを設定してください
genai.configure(api_key="YOUR_GEMINI_API_KEY")
model_gemini = genai.GenerativeModel('gemini-1.5-flash') # または 3-flash-preview

# 学習済みモデルの読み込み
lgb_model = lgb.Booster(model_file='boat_model_nirentan.txt')

# 特徴量リスト（学習時と同じ順番である必要があります）
# ※ここでは簡略化していますが、実際の学習時のリストを使用してください
FEATURES = ['jcd', 'rno', 'wind', 'wr_1_vs_avg', ...] 

# ==========================================
# 🔮 Gemini 審判関数
# ==========================================
def gemini_judgment(prob, odds, combinations):
    ev = prob * odds
    
    prompt = f"""
    あなたは競艇投資のプロフェッショナルなアドバイザーです。
    以下のデータに基づき、このレースに「投資すべきか」を判断してください。

    【データ】
    - 買い目: {combinations}
    - AIが算出した的中率: {prob*100:.1f}%
    - 現在のオッズ: {odds:.1f}倍
    - 計算された期待値: {ev:.2f}

    【判断基準】
    - 期待値(EV)が1.2未満なら「見送り」が基本。
    - 的中率が20%未満の場合、期待値が高くてもハイリスクとして警告すること。
    - 期待値が1.5を超える場合は「勝負レース」として推奨すること。

    返答は以下の形式で短く回答してください：
    【判定】買い / 見送り / 強気
    【理由】なぜそう判断したか
    【推奨金額】1万円の軍資金がある場合、いくら張るべきか（ケリー基準などを考慮）
    """
    
    response = model_gemini.generate_content(prompt)
    return response.text

# ==========================================
# 🚀 実行フロー
# ==========================================
def main():
    print("🚤 --- Gemini 競艇審判システム ---")
    
    # 1. 今日のレースデータを入力（実際には自動取得を想定）
    # ※ここでは例として1号艇の勝率などをダミー入力
    input_data = [24, 10, 2.0, 1.15, ...] # 会場24(大村), 10R, 風2m...
    
    # 2. AIで確率を算出
    probs = lgb_model.predict([input_data])[0]
    best_idx = np.argmax(probs)
    ai_prob = probs[best_idx]
    
    # 3. オッズの入力
    print(f"\nAI予測: 1-2 の的中率は {ai_prob*100:.1f}% です。")
    current_odds = float(input("現在のオッズを入力してください（例: 3.5）: "))
    
    # 4. Geminiの判断
    print("\n🤖 Geminiが判断中...")
    result = gemini_judgment(ai_prob, current_odds, "1-2")
    
    print("\n" + "="*30)
    print(result)
    print("="*30)

if __name__ == "__main__":
    main()