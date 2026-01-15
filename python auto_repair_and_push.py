import os
import shutil
import subprocess
import lightgbm as lgb

# === 設定エリア ===
REPO_URL = "https://github.com/sarusarugame-commits/kyouteigachi"
MODEL_FILE = "boat_model_nirentan.txt"
CSV_FILE = "FINAL_FULL_DATA_2025_FIXED.csv"

def run_cmd(cmd):
    print(f"Executing: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def main():
    # 1. モデルの軽量化変換 (Text -> Binary)
    # これにより100MBを切り、LFSなしでもプッシュできるサイズになる可能性があります
    print("📦 モデルを軽量なバイナリ形式に変換・圧縮中...")
    bst = lgb.Booster(model_file=MODEL_FILE)
    # バイナリ形式で上書き保存（精度は変わりません）
    bst.save_model(MODEL_FILE) 
    
    # 2. 古いGit履歴の削除 (タイムアウトの原因を排除)
    if os.path.exists(".git"):
        print("💥 古い履歴を削除中...")
        shutil.rmtree(".git")

    # 3. Git初期化
    run_cmd("git init")
    run_cmd("git lfs install")
    run_cmd(f'git lfs track "{MODEL_FILE}"')
    
    # 4. .gitignore作成 (CSVを除外)
    with open(".gitignore", "w") as f:
        f.write(f"{CSV_FILE}\n*.csv\n.venv/\n__pycache__/\n")

    # 5. コミット
    run_cmd("git add .")
    run_cmd("git add .gitattributes")
    run_cmd('git commit -m "Auto: 圧縮モデルとプログラム一式をアップロード"')

    # 6. プッシュ
    run_cmd("git branch -M main")
    run_cmd(f"git remote add origin {REPO_URL}")
    run_cmd("git config http.postBuffer 524288000")
    
    print("🚀 GitHubへプッシュを開始します（圧縮済みなので速いです）...")
    run_cmd("git push -u origin main --force")

    print("\n✅ すべて完了しました！GitHubを確認してください。")

if __name__ == "__main__":
    main()