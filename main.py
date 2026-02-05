ご要望に合わせて、バグ（結果判定のキー不一致）を修正し、かつ**「見送りの理由（スコア不足の詳細）」**をログに出力する機能を追加して書き直しました。

ファイル構成は元の通り3つ（main.py, predict_boat.py, scraper.py）に分けるのが適切ですので、それぞれの完成形を記述します。

変更点の概要

見送りログの強化:

自信度不足: AIの予測スコア（確率）が基準に届かなかった場合、そのスコアと基準値を表示。

期待値(EV)不足: 確率は十分だがオッズが低く、期待値が基準に届かなかった場合、「最大EV vs 基準EV」を表示。

バグ修正:

main.py と scraper.py 間でのキー（combo_2tなど）の不一致を解消。これで的中判定が正常に動きます。

ロジック改善:

filter_and_sort_bets で期待値(EV)が高い順にソートするように変更（以前は確率順だったため、高配当のチャンスを逃す可能性があった）。

1. main.py

実行用のメインファイルです。

code
Python
download
content_copy
expand_less
import os
import datetime
import time
import sqlite3
import concurrent.futures
import threading
import sys
import requests as std_requests
import json

# scraper, predict_boat は同じフォルダに配置してください
from scraper import scrape_race_data, get_session, get_odds_map, get_odds_2t, scrape_result
from predict_boat import predict_race, attach_reason, load_model, filter_and_sort_bets, CONF_THRESH_3T, CONF_THRESH_2T

DB_FILE = "race_data.db"
PLACE_NAMES = {i: n for i, n in enumerate(["","桐生","戸田","江戸川","平和島","多摩川","浜名湖","蒲郡","常滑","津","三国","びわこ","住之江","尼崎","鳴門","丸亀","児島","宮島","徳山","下関","若松","芦屋","福岡","唐津","大村"])}
JST = datetime.timezone(datetime.timedelta(hours=9), 'JST')

sys.stdout.reconfigure(encoding='utf-8')

DB_LOCK = threading.Lock()
STATS = {"scanned": 0, "hits": 0, "errors": 0, "skipped": 0, "vetted": 0, "waiting": 0}
STATS_LOCK = threading.Lock()
FINISHED_RACES = set()
FINISHED_RACES_LOCK = threading.Lock()

def log(msg):
    print(f"[{datetime.datetime.now(JST).strftime('%H:%M:%S')}] {msg}", flush=True)

def error_log(msg):
    print(f"[{datetime.datetime.now(JST).strftime('%H:%M:%S')}] ❌ {msg}", file=sys.stderr, flush=True)

def send_discord(content):
    url = os.environ.get("DISCORD_WEBHOOK_URL")
    if not url: return
    try:
        std_requests.post(url, json={"content": content}, timeout=10)
    except: pass

def init_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS history (
            race_id TEXT PRIMARY KEY,
            date TEXT,
            place TEXT,
            race_no INTEGER,
            predict_combo TEXT,
            status TEXT,
            profit INTEGER,
            odds REAL,
            prob REAL,
            ev REAL,
            comment TEXT,
            ticket_type TEXT
        )
    """)
    conn.close()

def report_worker(stop_event):
    log("ℹ️ レポート監視スレッド起動 (2連単/3連単 両対応)")
    while not stop_event.is_set():
        try:
            with DB_LOCK:
                conn = sqlite3.connect(DB_FILE)
                conn.row_factory = sqlite3.Row
                pending = conn.execute("SELECT * FROM history WHERE status='PENDING'").fetchall()
                sess = get_session()
                
                for p in pending:
                    try:
                        # race_id形式: YYYYMMDD_JCD_RNO_COMBO_TYPE
                        parts = p['race_id'].split('_')
                        jcd = int(parts[1])
                    except: continue
                    
                    res = scrape_result(sess, jcd, p['race_no'], p['date'])
                    if not res: continue

                    combo = p['predict_combo']
                    ticket_type = p['ticket_type'] # '2t' or '3t'
                    
                    # 修正: scraper.pyのキーに合わせて取得
                    if ticket_type == '2t':
                        result_str = res.get('combo_2t', '未確定')
                        payout = res.get('payout_2t', 0)
                    else:
                        result_str = res.get('combo_3t', '未確定')
                        payout = res.get('payout_3t', 0)
                    
                    if result_str != "未確定" and result_str is not None:
                        # 的中判定
                        is_hit = (result_str == combo)
                        profit = payout - 100 if is_hit else -100
                        
                        conn.execute("UPDATE history SET status='FINISHED', profit=? WHERE race_id=?", (profit, p['race_id']))
                        conn.commit()

                        if is_hit:
                            today_str = p['date']
                            total_profit = conn.execute("SELECT SUM(profit) FROM history WHERE date=? AND status='FINISHED'", (today_str,)).fetchone()[0]
                            if total_profit is None: total_profit = 0

                            msg = (
                                f"🎯 **{p['place']}{p['race_no']}R** 的中！({ticket_type.upper()})\n"
                                f"買い目: {combo} ({p['odds']}倍)\n"
                                f"払戻: {payout:,}円 (収支: +{profit:,}円)\n"
                                f"📅 本日トータル: {total_profit:+,}円"
                            )
                            log(f"🎯 的中: {p['place']}{p['race_no']}R ({combo}) +{profit}円")
                            send_discord(msg)
                conn.close()
        except Exception as e:
            error_log(f"レポート監視エラー: {e}")
        time.sleep(120)

def process_race(jcd, rno, today):
    with FINISHED_RACES_LOCK:
        if (jcd, rno) in FINISHED_RACES: return

    sess = get_session()
    place = PLACE_NAMES.get(jcd, "不明")
    
    try:
        raw, error = scrape_race_data(sess, jcd, rno, today)
    except:
        with STATS_LOCK: STATS["errors"] += 1
        return

    if error != "OK" or not raw: return

    # 1. 予測実行 (会場フィルタリング含む)
    try:
        # candidates: 候補リスト
        # max_conf: AIの最大自信度(確率)
        # is_target: 戦略対象の会場かどうか
        candidates, max_conf, is_target = predict_race(raw)
    except Exception as e:
        error_log(f"予測エラー {place}{rno}R: {e}")
        with STATS_LOCK: STATS["errors"] += 1
        return

    if not is_target: return

    # --- 見送り理由ログ: 自信度不足 ---
    if not candidates:
        # 3Tか2Tかによって閾値の表示を変える（簡易的に3T基準で表示、または高い方）
        thresh_display = max(CONF_THRESH_3T, CONF_THRESH_2T)
        if max_conf > 0:
            log(f"👀 [見送り] {place}{rno}R: 自信度不足 (AIスコア:{max_conf:.2f} < 基準:{thresh_display})")
        with STATS_LOCK: STATS["vetted"] += 1
        return

    # 2. オッズ取得
    odds_2t, odds_3t = {}, {}
    has_2t = any(c['type'] == '2t' for c in candidates)
    has_3t = any(c['type'] == '3t' for c in candidates)
    
    try:
        if has_2t: odds_2t = get_odds_2t(sess, jcd, rno, today)
        if has_3t: odds_3t = get_odds_map(sess, jcd, rno, today)
    except Exception: pass

    # 3. EVフィルタリング
    try:
        final_bets, max_ev, current_thresh = filter_and_sort_bets(candidates, odds_2t, odds_3t, jcd)
    except: return

    # --- 見送り理由ログ: 期待値(EV)不足 ---
    if not final_bets:
        # 候補はあったが、オッズと掛け合わせたら期待値が足りなかった場合
        if max_ev > 0:
            log(f"📉 [見送り] {place}{rno}R: 期待値不足 (最大EV:{max_ev:.2f} < 基準:{current_thresh})")
        else:
            log(f"📉 [見送り] {place}{rno}R: オッズ取得失敗または有効オッズなし")
        
        with STATS_LOCK: STATS["vetted"] += 1
        return

    # 4. 時間管理
    deadline_str = raw.get('deadline_time')
    if deadline_str:
        try:
            now = datetime.datetime.now(JST)
            h, m = map(int, deadline_str.split(':'))
            deadline_dt = now.replace(hour=h, minute=m, second=0, microsecond=0)
            
            if now > (deadline_dt + datetime.timedelta(minutes=1)):
                with FINISHED_RACES_LOCK: FINISHED_RACES.add((jcd, rno))
                with STATS_LOCK: STATS["skipped"] += 1
                return

            delta = deadline_dt - now
            if delta.total_seconds() > 1200: # 20分前
                with STATS_LOCK: STATS["waiting"] += 1
                return
        except: pass

    # 5. 解説生成
    try:
        attach_reason(final_bets, raw, {})
    except Exception: pass

    # 6. DB保存 & 通知
    with STATS_LOCK: STATS["scanned"] += 1
    with DB_LOCK:
        conn = sqlite3.connect(DB_FILE)
        for p in final_bets:
            combo = p['combo']
            t_type = p['type']
            race_id = f"{today}_{jcd}_{rno}_{combo}_{t_type}"
            
            if conn.execute("SELECT 1 FROM history WHERE race_id=?", (race_id,)).fetchone(): continue

            prob = float(p.get('prob', 0))
            reason = p.get('reason', '解説取得失敗')
            odds_val = p.get('odds', 0.0)
            ev_val = p.get('ev', 0.0)
            
            log(f"🔥 [HIT] {place}{rno}R ({t_type.upper()}) -> {combo} ({odds_val}倍 EV:{ev_val:.2f})")
            
            odds_url = f"https://www.boatrace.jp/owpc/pc/race/odds{'2t' if t_type=='2t' else '3t'}?rno={rno}&jcd={jcd:02d}&hd={today}"

            msg = (
                f"🔥 **{place}{rno}R** {t_type.upper()}激アツ\n"
                f"🎯 買い目: **{combo}**\n"
                f"📊 確率: **{prob}%** / オッズ: **{odds_val}倍**\n"
                f"💎 期待値: **{ev_val:.2f}**\n"
                f"📝 AI寸評: {reason}\n"
                f"🔗 [オッズ確認]({odds_url})"
            )
            
            conn.execute(
                "INSERT INTO history VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (race_id, today, place, rno, combo, 'PENDING', 0, odds_val, prob, ev_val, reason, t_type)
            )
            conn.commit()
            send_discord(msg)
            with STATS_LOCK: STATS["hits"] += 1
        conn.close()

def main():
    log(f"🚀 ハイブリッドAI Bot (ROI130% & 黄金律) 起動")
    
    try:
        load_model()
        log("✅ AIモデル(2T/3T) 読み込み完了")
    except Exception as e:
        error_log(f"FATAL: モデル読み込みエラー: {e}")
        sys.exit(1)

    init_db()
    stop_event = threading.Event()
    t = threading.Thread(target=report_worker, args=(stop_event,), daemon=True)
    t.start()
    
    start_time = time.time()
    MAX_RUNTIME = 21600 # 6時間
    
    while True:
        if time.time() - start_time > MAX_RUNTIME:
            log("🔄 稼働時間上限のため終了")
            break
        
        now = datetime.datetime.now(JST)
        if now.hour == 23 and now.minute >= 55: break
            
        today = now.strftime('%Y%m%d')
        
        # 統計リセット
        with STATS_LOCK:
            STATS["scanned"] = 0; STATS["hits"] = 0; STATS["errors"] = 0
            STATS["skipped"] = 0; STATS["vetted"] = 0; STATS["waiting"] = 0

        log(f"🔍 スキャン開始 ({today})...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
            for rno in range(1, 13):
                for jcd in range(1, 25):
                    ex.submit(process_race, jcd, rno, today)

        log(f"🏁 サイクル完了: 購入={STATS['hits']}, 見送り={STATS['vetted']}, 待機={STATS['waiting']}, 締切={STATS['skipped']}")
        time.sleep(60)

    stop_event.set()

if __name__ == "__main__":
    main()
2. predict_boat.py

予測ロジックです。filter_and_sort_betsをEV順ソートに修正し、モデルロード部分を整理しました。

code
Python
download
content_copy
expand_less
import pandas as pd
import numpy as np
import lightgbm as lgb
import os
from itertools import permutations

# ==========================================
# ⚙️ 設定: 券種別・完全独立パラメータ
# ==========================================

# --- 三連単 (3T) 黄金律設定 ---
MIN_PROB_3T = 0.03
ODDS_CAP_3T = 40.0
MAX_BETS_3T = 6
CONF_THRESH_3T = 0.20
STRATEGY_3T = {
    2: 2.0, 3: 1.2, 5: 2.0, 6: 1.6, 8: 1.8, 9: 1.4, 10: 1.3,
    11: 2.5, 13: 1.6, 14: 1.6, 16: 1.5, 19: 1.3, 20: 2.0,
    22: 1.2, 23: 1.5, 24: 1.5
}

# --- 二連単 (2T) ROI 130% 厳選設定 ---
MIN_PROB_2T = 0.01
ODDS_CAP_2T = 100.0
MAX_BETS_2T = 8
CONF_THRESH_2T = 0.0
STRATEGY_2T = {
    8: 4.0, 10: 4.0, 16: 3.0, 21: 2.5
}

# ==========================================
# 🤖 Groq (OpenAI Client Wrapper) 設定
# ==========================================
OPENAI_AVAILABLE = False
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    pass

_GROQ_CLIENT = None

def get_groq_client():
    global _GROQ_CLIENT
    if not OPENAI_AVAILABLE: return None
    if _GROQ_CLIENT is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key: return None
        try:
            _GROQ_CLIENT = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=api_key,
                max_retries=3, 
                timeout=20.0
            )
        except: return None
    return _GROQ_CLIENT

# --- モデル管理 ---
MODELS = {'3t': None, '2t': None}

def load_model():
    # 3Tモデル
    if MODELS['3t'] is None:
        if os.path.exists("boatrace_model.txt"):
            MODELS['3t'] = lgb.Booster(model_file="boatrace_model.txt")
        elif os.path.exists("boat_race_model_3t.txt"):
            MODELS['3t'] = lgb.Booster(model_file="boat_race_model_3t.txt")
    
    # 2Tモデル
    if MODELS['2t'] is None:
        if os.path.exists("boatrace_model_2t.txt"):
            MODELS['2t'] = lgb.Booster(model_file="boatrace_model_2t.txt")
        
    return MODELS

def to_float(val):
    try:
        if val is None or val == "": return 0.0
        return float(val)
    except: return 0.0

# ==========================================
# 🔮 1. 候補出し (3T / 2T 独立判定)
# ==========================================
def predict_race(raw):
    """
    戻り値: (候補リスト, 最大自信度, 戦略対象フラグ)
    """
    load_model()
    jcd = int(raw.get('jcd', 0))
    use_3t = jcd in STRATEGY_3T
    use_2t = jcd in STRATEGY_2T
    
    if not use_3t and not use_2t:
        return [], 0.0, False

    # 特徴量生成
    rows = []
    ex_list = []
    wind = to_float(raw.get('wind', 0.0))
    for i in range(1, 7):
        s = str(i)
        val_ex = to_float(raw.get(f'ex{s}', 0))
        ex_list.append(val_ex)
        rows.append({
            'jcd': jcd, 'wind': wind, 'boat_no': i,
            'pid': raw.get(f'pid{s}', 0), 
            'wr': to_float(raw.get(f'wr{s}', 0)),
            'mo': to_float(raw.get(f'mo{s}', 0)), 
            'ex': val_ex,
            'st': to_float(raw.get(f'st{s}', 0.20)), 
            'f': to_float(raw.get(f'f{s}', 0)),
        })
    
    if sum(ex_list) == 0: return [], 0.0, True

    df = pd.DataFrame(rows)
    for col in ['wr', 'mo', 'ex', 'st']:
        m, s = df[col].mean(), df[col].std()
        df[f'{col}_z'] = (df[col] - m) / (s if s != 0 else 1e-6)

    df['jcd'] = df['jcd'].astype('category')
    df['pid'] = df['pid'].astype('category')
    features = ['jcd', 'boat_no', 'pid', 'wind', 'wr', 'mo', 'ex', 'st', 'f', 'wr_z', 'mo_z', 'ex_z', 'st_z']
    
    candidates = []
    max_p1 = 0.0
    b = df['boat_no'].values

    # --- 三連単 判定 ---
    if MODELS['3t'] and use_3t:
        p = MODELS['3t'].predict(df[features])
        p1, p2, p3 = p[:, 0], p[:, 1], p[:, 2]
        current_max = max(p1)
        max_p1 = max(max_p1, current_max)
        
        if current_max >= CONF_THRESH_3T:
            for i, j, k in permutations(range(6), 3):
                prob = p1[i] * p2[j] * p3[k]
                if prob >= MIN_PROB_3T:
                    candidates.append({
                        'combo': f"{b[i]}-{b[j]}-{b[k]}", 
                        'raw_prob': prob, 
                        'prob': round(prob * 100, 1),
                        'type': '3t'
                    })

    # --- 二連単 判定 ---
    if MODELS['2t'] and use_2t:
        p_2t = MODELS['2t'].predict(df[features])
        p1_2, p2_2 = p_2t[:, 0], p_2t[:, 1]
        current_max = max(p1_2)
        max_p1 = max(max_p1, current_max)

        if current_max >= CONF_THRESH_2T:
            for i, j in permutations(range(6), 2):
                prob = p1_2[i] * p2_2[j]
                if prob >= MIN_PROB_2T:
                    candidates.append({
                        'combo': f"{b[i]}-{b[j]}", 
                        'raw_prob': prob, 
                        'prob': round(prob * 100, 1),
                        'type': '2t'
                    })

    # 確率順にソート (EV計算前の一時ソート)
    candidates.sort(key=lambda x: x['raw_prob'], reverse=True)
    return candidates, max_p1, True

# ==========================================
# 💰 2. EVフィルタ
# ==========================================
def filter_and_sort_bets(candidates, odds_2t, odds_3t, jcd):
    final_2t, final_3t = [], []
    max_ev = 0.0
    
    # 戦略閾値の取得 (3T優先、なければ2T。ログ用)
    strategy_thresh = STRATEGY_3T.get(jcd) if jcd in STRATEGY_3T else STRATEGY_2T.get(jcd, 99.0)

    for c in candidates:
        combo = c['combo']
        prob = c['raw_prob']
        ev = 0.0
        
        if c['type'] == '2t':
            real_o = odds_2t.get(combo, 0.0)
            if real_o > 0:
                ev = prob * min(real_o, ODDS_CAP_2T)
                if ev > max_ev: max_ev = ev
                if ev >= STRATEGY_2T.get(jcd, 99.0):
                    c.update({'odds': real_o, 'ev': ev})
                    final_2t.append(c)
        else:
            real_o = odds_3t.get(combo, 0.0)
            if real_o > 0:
                ev = prob * min(real_o, ODDS_CAP_3T)
                if ev > max_ev: max_ev = ev
                if ev >= STRATEGY_3T.get(jcd, 99.0):
                    c.update({'odds': real_o, 'ev': ev})
                    final_3t.append(c)
    
    # 修正: 期待値(EV)が高い順にソートし直す
    final_2t.sort(key=lambda x: x['ev'], reverse=True)
    final_3t.sort(key=lambda x: x['ev'], reverse=True)
            
    return final_2t[:MAX_BETS_2T] + final_3t[:MAX_BETS_3T], max_ev, strategy_thresh

# ==========================================
# 📝 3. 解説生成
# ==========================================
def generate_batch_reasons(jcd, bets_info, raw_data):
    client = get_groq_client()
    if not client: return {}
    
    players_info = ""
    for i in range(1, 7):
        players_info += f"{i}号艇:勝率{raw_data.get(f'wr{i}',0)} "

    bets_text = ""
    for b in bets_info:
        bets_text += f"- {b['combo']}({b['type'].upper()}): 確率{b['prob']}% オッズ{b['odds']} (期待値{b['ev']:.2f})\n"

    prompt = f"""
    ボートレース予想家として、以下の{jcd}場の買い目を解説せよ。
    [選手] {players_info}
    [買い目] {bets_text}
    【指示】
    各買い目について、なぜチャンスなのか 30文字以内 でコメント。
    必ず 【勝負】 か 【見送り】 で始めること。
    """
    
    try:
        chat = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile", temperature=0.7, max_tokens=400
        )
        text = chat.choices[0].message.content
        comments = {}
        for line in text.split('\n'):
            if ':' in line:
                p = line.split(':', 1)
                comments[p[0].strip()] = p[1].strip()
        return comments
    except: return {}

def attach_reason(results, raw, odds_map=None):
    if not results: return
    jcd = raw.get('jcd', 0)
    ai_comments = generate_batch_reasons(jcd, results, raw)
    for item in results:
        ai_msg = ai_comments.get(item['combo'])
        if ai_msg:
            item['reason'] = f"{ai_msg} (EV:{item['ev']:.2f})"
        else:
            item['reason'] = f"【勝負】AI推奨 (EV:{item['ev']:.2f})"
3. scraper.py

スクレイピング用。scrape_resultの修正を含みます。

code
Python
download
content_copy
expand_less
from curl_cffi import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import re
import unicodedata
import warnings

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

def clean_text(text):
    if not text: return ""
    text = unicodedata.normalize('NFKC', str(text))
    return text.replace("\n", "").replace("\r", "").replace("¥", "").replace(",", "").strip()

def get_session():
    # Chrome 120 の指紋を模倣
    return requests.Session(impersonate="chrome120")

def get_soup(session, url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://www.boatrace.jp/",
            "Accept-Language": "ja,en-US;q=0.9,en;q=0.8"
        }
        res = session.get(url, headers=headers, timeout=15)
        
        if "データがありません" in res.text: return None, "NO_RACE"
        if res.status_code == 404: return None, "NO_RACE"
        if res.status_code != 200: return None, "HTTP_ERROR"
        if len(res.content) < 500: return None, "SMALL_CONTENT"
        
        return BeautifulSoup(res.content, 'lxml'), "OK"
    except Exception as e:
        return None, f"EXCEPTION_{e}"

def extract_deadline(soup, rno):
    if not soup: return None
    try:
        candidates = soup.find_all(['th', 'td'], string=re.compile(r"締切|予定"))
        for tag in candidates:
            parent_row = tag.find_parent("tr")
            if not parent_row: continue
            cells = parent_row.find_all(['td', 'th'])
            time_cells = []
            for cell in cells:
                txt = clean_text(cell.text)
                if re.search(r"\d{1,2}:\d{2}", txt):
                    time_cells.append(txt)
            
            if len(time_cells) >= 10:
                if 1 <= rno <= len(time_cells):
                    target_time = time_cells[rno - 1]
                    m = re.search(r"(\d{1,2}:\d{2})", target_time)
                    if m: return m.group(1).zfill(5)
            
            next_tag = tag.find_next_sibling(['td', 'th'])
            if next_tag:
                text = clean_text(next_tag.text)
                m = re.search(r"(\d{1,2}:\d{2})", text)
                if m: return m.group(1).zfill(5)
            
            text = clean_text(tag.text)
            m = re.search(r"(\d{1,2}:\d{2})", text)
            if m: return m.group(1).zfill(5)
    except Exception: pass
    return None

def scrape_race_data(session, jcd, rno, date_str):
    base_url = "https://www.boatrace.jp/owpc/pc/race"
    
    url_before = f"{base_url}/beforeinfo?rno={rno}&jcd={jcd:02d}&hd={date_str}"
    soup_before, stat_b = get_soup(session, url_before)
    
    url_list = f"{base_url}/racelist?rno={rno}&jcd={jcd:02d}&hd={date_str}"
    soup_list, stat_l = get_soup(session, url_list)

    if stat_b == "NO_RACE" or stat_l == "NO_RACE":
        return None, "NO_RACE"

    if not soup_before and not soup_list: 
        return None, f"FETCH_ERR({stat_b}/{stat_l})"

    row = {
        'date': int(date_str), 'jcd': jcd, 'rno': rno, 'wind': 0.0,
        'deadline_time': None
    }
    
    for i in range(1, 7):
        row[f'pid{i}'] = 0
        row[f'wr{i}'] = 0.0
        row[f'mo{i}'] = 0.0
        row[f'ex{i}'] = 0.0
        row[f'f{i}'] = 0
        row[f'st{i}'] = 0.20

    row['deadline_time'] = extract_deadline(soup_before, rno)
    if not row['deadline_time']:
        row['deadline_time'] = extract_deadline(soup_list, rno)
        
    if soup_before:
        try:
            wind_unit = soup_before.select_one(".is-windDirection")
            if wind_unit:
                wind_data = wind_unit.select_one(".weather1_bodyUnitLabelData")
                if wind_data:
                    w_txt = clean_text(wind_data.text)
                    m = re.search(r"(\d+)", w_txt)
                    if m: row['wind'] = float(m.group(1))
            if row['wind'] == 0.0:
                 m = re.search(r"風.*?(\d+)m", soup_before.text)
                 if m: row['wind'] = float(m.group(1))
        except: pass

    for i in range(1, 7):
        if soup_before:
            try:
                boat_td = soup_before.select_one(f"td.is-boatColor{i}")
                if boat_td:
                    tr = boat_td.find_parent("tr")
                    if tr:
                        text_all = clean_text(tr.text)
                        matches = re.findall(r"(6\.\d{2}|7\.[0-4]\d)", text_all)
                        if matches: row[f'ex{i}'] = float(matches[-1])
            except: pass
            
        if soup_list:
            try:
                tbodies = soup_list.select("tbody.is-fs12")
                if len(tbodies) >= i:
                    tbody = tbodies[i-1]
                    txt_all = clean_text(tbody.text)
                    
                    pid_match = re.search(r"([2-5]\d{3})", txt_all)
                    if pid_match: row[f'pid{i}'] = int(pid_match.group(1))
                    
                    wr_matches = re.findall(r"(\d\.\d{2})", txt_all)
                    for val_str in wr_matches:
                        val = float(val_str)
                        if 1.0 <= val <= 9.99: 
                            row[f'wr{i}'] = val
                            break
                            
                    mo_matches = re.findall(r"(\d{2}\.\d{2})", txt_all)
                    for m_val in mo_matches:
                        if 10.0 <= float(m_val) <= 99.9: 
                            row[f'mo{i}'] = float(m_val)
                            break
                            
                    st_match = re.search(r"(0\.\d{2})", txt_all)
                    if st_match: row[f'st{i}'] = float(st_match.group(1))
                    
                    f_match = re.search(r"F(\d+)", txt_all)
                    if f_match: row[f'f{i}'] = int(f_match.group(1))
            except: pass
            
    return row, "OK"

def get_odds_map(session, jcd, rno, date_str):
    url = f"https://www.boatrace.jp/owpc/pc/race/odds3t?rno={rno}&jcd={jcd:02d}&hd={date_str}"
    soup, _ = get_soup(session, url)
    if not soup: return {}

    odds_map = {}
    tables = soup.select("div.table1 table")
    
    for tbl in tables:
        if "3連単" not in tbl.text: continue
        tbody = tbl.select_one("tbody")
        if not tbody: continue
        rows = tbody.select("tr")
        rowspan_counters = [0] * 6
        current_2nd_boats = [0] * 6

        for tr in rows:
            tds = tr.select("td")
            col_cursor = 0
            for block_idx in range(6):
                if col_cursor >= len(tds): break
                current_1st = block_idx + 1 
                if rowspan_counters[block_idx] > 0:
                    if col_cursor + 1 >= len(tds): break
                    val_2nd = current_2nd_boats[block_idx]
                    txt_3rd = clean_text(tds[col_cursor].text)
                    txt_odds = clean_text(tds[col_cursor+1].text)
                    rowspan_counters[block_idx] -= 1
                    col_cursor += 2
                else:
                    if col_cursor + 2 >= len(tds): break
                    td_2nd = tds[col_cursor]
                    txt_2nd = clean_text(td_2nd.text)
                    rs = 1
                    if td_2nd.has_attr("rowspan"):
                        try: rs = int(td_2nd["rowspan"])
                        except: rs = 1
                    rowspan_counters[block_idx] = rs - 1
                    try: val_2nd = int(txt_2nd)
                    except: val_2nd = 0
                    current_2nd_boats[block_idx] = val_2nd
                    txt_3rd = clean_text(tds[col_cursor+1].text)
                    txt_odds = clean_text(tds[col_cursor+2].text)
                    col_cursor += 3

                try:
                    if val_2nd > 0 and txt_3rd.isdigit():
                        key = f"{current_1st}-{val_2nd}-{txt_3rd}"
                        odds_val = float(txt_odds)
                        if odds_val > 0: odds_map[key] = odds_val
                except: continue
    return odds_map

def get_odds_2t(session, jcd, rno, date_str):
    url = f"https://www.boatrace.jp/owpc/pc/race/odds2tf?rno={rno}&jcd={jcd:02d}&hd={date_str}"
    soup, _ = get_soup(session, url)
    if not soup: return {}
    
    odds_map = {}
    tables = soup.select("table")
    
    for tbl in tables:
        txt = tbl.text
        if "2連単" not in txt and "２連単" not in txt: continue

        rows = tbl.select("tr")
        current_1st = 0
        
        for tr in rows:
            boat_num_icon = tr.select_one("div.numberSet1_number") 
            if boat_num_icon:
                try: current_1st = int(clean_text(boat_num_icon.text))
                except: pass
            
            text_cells = [clean_text(td.text) for td in tr.select("td")]
            for i in range(0, len(text_cells), 2):
                if i+1 >= len(text_cells): break
                try:
                    sec = int(text_cells[i])
                    odd = float(text_cells[i+1])
                    if current_1st != 0 and sec != 0:
                        odds_map[f"{current_1st}-{sec}"] = odd
                except: pass
    return odds_map

def scrape_result(session, jcd, rno, date_str):
    url = f"https://www.boatrace.jp/owpc/pc/race/raceresult?rno={rno}&jcd={jcd:02d}&hd={date_str}"
    soup, _ = get_soup(session, url)
    if not soup: return None
    
    # 初期値の設定
    res = {
        'combo_3t': None, 'payout_3t': 0,
        'combo_2t': None, 'payout_2t': 0
    }
    
    try:
        tables = soup.select("table.is-w495")
        for tbl in tables:
            # 3連単
            if "3連単" in tbl.text:
                rows = tbl.select("tr")
                for tr in rows:
                    if "3連単" in tr.text:
                        combo_node = tr.select(".numberSet1_number")
                        if combo_node:
                            nums = [c.text.strip() for c in combo_node]
                            res['combo_3t'] = "-".join(nums)
                        tds = tr.select("td")
                        for td in reversed(tds):
                            txt = clean_text(td.text).replace("¥","").replace(",","")
                            if txt.isdigit() and int(txt) >= 100:
                                res['payout_3t'] = int(txt); break
            
            # 2連単
            if "2連単" in tbl.text:
                rows = tbl.select("tr")
                for tr in rows
