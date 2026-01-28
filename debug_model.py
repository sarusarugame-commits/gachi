import joblib
import os
import sys
import lightgbm
import pandas
import numpy
import traceback

# scikit-learnも確認（LightGBMが依存している場合があるため）
try:
    import sklearn
    sklearn_version = sklearn.__version__
except ImportError:
    sklearn_version = "未インストール"

MODEL_FILE = 'ultimate_boat_model.pkl'

print("="*50)
print("🔍 環境・モデル診断ツール")
print("="*50)

# 1. バージョン確認
print("\n[1] ライブラリバージョン")
print(f"Python: {sys.version}")
print(f"Pandas: {pandas.__version__}")
print(f"Numpy: {numpy.__version__}")
print(f"LightGBM: {lightgbm.__version__}")
print(f"Joblib: {joblib.__version__}")
print(f"Scikit-learn: {sklearn_version}")

# 2. ファイル存在確認
print("\n[2] ファイル診断")
if os.path.exists(MODEL_FILE):
    size = os.path.getsize(MODEL_FILE)
    print(f"✅ ファイルが見つかりました: {MODEL_FILE}")
    print(f"📦 サイズ: {size / (1024*1024):.2f} MB")
    
    if size < 1000:
        print("⚠️ 警告: ファイルサイズが小さすぎます。Git LFSのポインタファイルの可能性があります。")
else:
    print(f"❌ ファイルが見つかりません: {MODEL_FILE}")
    print("   -> ファイル名が間違っているか、アップロードされていません。")
    sys.exit(1)

# 3. ロードテスト (詳細エラー表示)
print("\n[3] ロードテスト開始...")
try:
    model = joblib.load(MODEL_FILE)
    print("🎉 成功: モデルは正常に読み込めました！")
    print(f"   Type: {type(model)}")
    
    # 辞書型ならキーを表示
    if isinstance(model, dict):
        print(f"   Keys: {model.keys()}")
        
except Exception as e:
    print("💀 失敗: モデルの読み込みに失敗しました。")
    print("-" * 30)
    print(f"エラーメッセージ: {e}")
    print("-" * 30)
    print("詳細スタックトレース:")
    traceback.print_exc()
    print("-" * 30)
    print("【対策】")
    print("エラー内容に 'ModuleNotFoundError' がある場合 -> requirements.txt にそのライブラリを追加してください。")
    print("エラー内容に 'version mismatch' 系がある場合 -> 学習環境と実行環境のバージョンを揃えてください。")

print("="*50)
