import os
import zipfile
import subprocess
import shutil
import stat

# 作業ディレクトリの設定
target_dir = r"C:\Users\TAKUMA\競艇に勝つ\競艇データ"
os.chdir(target_dir)

REPO_URL = 'https://github.com/sarusarugame-commits/kyouteigachi'
MODEL_FILE = 'boat_model_nirentan.txt'
ZIP_MODEL = 'model.zip'
CHUNK_SIZE = 90 * 1024 * 1024  # 90MBごとに分割（GitHub制限回避）

def remove_readonly(func, path, excinfo):
    os.chmod(path, stat.S_IWRITE)
    func(path)

def run(c):
    print(f'Running: {c}')
    return subprocess.run(c, shell=True)

def main():
    # 1. モデルを圧縮
    if os.path.exists(MODEL_FILE):
        print(f'📦 {MODEL_FILE} を圧縮中...')
        with zipfile.ZipFile(ZIP_MODEL, 'w', zipfile.ZIP_DEFLATED) as f:
            f.write(MODEL_FILE)
    
    # 2. 圧縮ファイルを分割（161MB -> 85MB x 2ファイルなど）
    print(f'✂️ {ZIP_MODEL} を分割中...')
    with open(ZIP_MODEL, 'rb') as f:
        chunk_num = 1
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk: break
            with open(f'model_part_{chunk_num}', 'wb') as chunk_f:
                chunk_f.write(chunk)
            print(f'  -> model_part_{chunk_num} 作成')
            chunk_num += 1

    # 3. main.py の自動修正（サーバー上で分割ファイルを結合して解凍するコード）
    if os.path.exists('main.py'):
        with open('main.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 分割ファイルを合体させるコード
        join_code = (
            "import os, zipfile\n"
            "if not os.path.exists('boat_model_nirentan.txt'):\n"
            "    print('🧩 分割されたモデルを結合中...')\n"
            "    with open('recombined_model.zip', 'wb') as f_out:\n"
            "        for i in range(1, 10):\n"
            "            part = f'model_part_{i}'\n"
            "            if os.path.exists(part):\n"
            "                with open(part, 'rb') as f_in: f_out.write(f_in.read())\n"
            "    with zipfile.ZipFile('recombined_model.zip', 'r') as f: f.extractall()\n\n"
        )
        if 'recombined_model.zip' not in content:
            print("📝 main.py に結合・解凍コードを追加中...")
            with open('main.py', 'w', encoding='utf-8') as f:
                f.write(join_code + content)

    # 4. 古いGit履歴の強制削除
    if os.path.exists('.git'):
        run('rmdir /s /q .git')
        if os.path.exists('.git'): shutil.rmtree('.git', onerror=remove_readonly)

    # 5. 新規Git構築
    run('git init')
    
    # 巨大な生モデル、巨大なzip、CSVを無視（分割した model_part_* だけを送る）
    with open('.gitignore', 'w') as f:
        f.write(f'{MODEL_FILE}\n{ZIP_MODEL}\n*.csv\n.venv/\n__pycache__/\n')

    run('git add .')
    run('git commit -m "Final version with split model parts"')
    run('git branch -M main')
    run(f'git remote add origin {REPO_URL}')
    run('git config http.postBuffer 524288000')
    
    print('🚀 GitHubへ送信中（各ファイル100MB以下なので確実に通ります）...')
    result = run('git push -u origin main --force')
    
    if result.returncode == 0:
        print('\n✨ 大成功！すべての制限を突破してプッシュが完了しました。')
    else:
        print('\n❌ 失敗。ネット接続やリポジトリURLを確認してください。')

if __name__ == "__main__":
    main()