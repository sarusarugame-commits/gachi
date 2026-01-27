import os
import datetime
import time
import requests
import sqlite3
import sys
import logging

# 自作モジュール
from scraper import scrape_race_data, scrape_odds, scrape_result, get_session
from predict_boat import predict_race

# ==========================================
# 📝 ログ設定 (画面とファイルの両方に出す)
# ==========================================
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),      # 画面に出す
        logging.FileHandler("debug_log.txt", mode='w', encoding='utf-8') # ファイルに書く
    ]
)
logger = logging.getLogger(__name__)

DB_FILE = "race_data.db"
BET_AMOUNT = 1000 
PLACE_NAMES = {i: n for i, n in enumerate(["","桐生","戸田","江戸川","平和島","多摩川","浜名湖","蒲郡","常滑","津","三国","びわこ","住之江","尼崎","鳴門","丸亀","児島","宮島","徳山","下関","若松","芦屋","福岡","唐津","大村"])}
JST = datetime.timezone(datetime.timedelta(hours=9), 'JST')

def send_discord(content):
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if url: 
        try:
            requests.post(url, json={"content": content}, timeout=10)
        except Exception as e:
            logger.error(f"Discord送信エラー: {e}")

def init_db():
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.execute("CREATE TABLE IF NOT EXISTS history (race_id TEXT PRIMARY KEY, date TEXT, place TEXT, race_no INTEGER, predict_combo TEXT, status TEXT, profit INTEGER)")
        conn.close()
        logger.info("DB初期化完了")
    except Exception as e:
        logger.critical(f"DB初期化失敗: {e}")
        sys.exit(1)

def process_race_sequential(jcd, rno, today):
    """並列ではなく、1つずつ確実に処理する"""
    place_name = PLACE_NAMES.get(jcd, f"場{jcd}")
    logger.debug(f"🔍 [Check] {place_name}{rno}R データ取得開始...")

    try:
        sess = get_session()
        raw = scrape_race_data(sess, jcd, rno, today)
    except Exception as e:
        logger.error(f"❌ {place_name}{rno}R スクレイピングで例外発生: {e}")
        return

    if not raw:
        # 情報がない場合はDEBUGレベルでひっそりと（ログが埋まるので）
        # logger.debug(f"💨 {place_name}{rno}R 情報なし(スキップ)")
        return

    # データが取れたらINFOで表示
    logger.info(f"✅ {place_name}{rno}R 取得成功 | 締切:{raw.get('deadline_time')} | 1号艇勝率:{raw.get('wr1')}")

    # 安全装置解除: 0でも突っ込む
    try:
        preds = predict_race(raw)
    except Exception as e:
        logger.error(f"❌ {place_name}{rno}R 予測ロジックでエラー: {e}")
        return

    if not preds:
        return

    conn = sqlite3.connect(DB_FILE)
    for p in preds:
        race_id = f"{today}_{jcd}_{rno}_{p['combo']}"
        exists = conn.execute("SELECT 1 FROM history WHERE race_id=?", (race_id,)).fetchone()
        
        if not exists:
            logger.info(f"🔥 【激熱発見】 {place_name}{rno}R -> {p['combo']}")
            
            conn.execute("INSERT INTO history VALUES (?,?,?,?,?,?,?)", (race_id, today, place_name, rno, p['combo'], 'PENDING', 0))
            conn.commit()
            send_discord(f"🔥 **{place_name}{rno}R** 推奨:[{p['type']}] {p['combo']} (実績期待値:{p['profit']}円)")
    conn.close()

def main():
    logger.info("🚀 最強AI Bot (シングルスレッド・ファイルログ版) 起動")
    init_db()
    
    # ループ開始
    while True:
        today = datetime.datetime.now(JST).strftime('%Y%m%d')
        logger.info(f"⚡ 巡回開始: {datetime.datetime.now(JST).strftime('%H:%M:%S')}")
        
        # 全24場 x 12R を「順番に」回す (遅いが確実)
        for jcd in range(1, 25):
            for rno in range(1, 13):
                process_race_sequential(jcd, rno, today)
                # サーバー負荷軽減のためごく短時間待つ
                time.sleep(0.1)

        logger.info("💤 巡回終了。5分待機します...")
        time.sleep(300)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"💀 メインプロセスがクラッシュしました: {e}")
        # エラー詳細をファイルに吐く
        import traceback
        with open("crash_log.txt", "w") as f:
            f.write(traceback.format_exc())
