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
    # Chrome 120 偽装 (ブロック回避)
    return requests.Session(impersonate="chrome120")

def get_soup(session, url):
    try:
        res = session.get(url, timeout=10)
        if res.status_code != 200: return None
        if len(res.content) < 5000: return None # Block check
        if "データがありません" in res.text: return None
        return BeautifulSoup(res.content, 'lxml')
    except: return None

def scrape_race_data(session, jcd, rno, date_str):
    base_url = "https://www.boatrace.jp/owpc/pc/race"
    
    # 3ページ全てにアクセス
    soup_before = get_soup(session, f"{base_url}/beforeinfo?rno={rno}&jcd={jcd:02d}&hd={date_str}")
    soup_list = get_soup(session, f"{base_url}/racelist?rno={rno}&jcd={jcd:02d}&hd={date_str}")
    soup_res = get_soup(session, f"{base_url}/raceresult?rno={rno}&jcd={jcd:02d}&hd={date_str}")

    if not soup_before or not soup_list:
        # 最低限、出走表がないと話にならない
        return None, "NO_DATA"

    # --- 1. 全42項目の初期化 (指定された順序) ---
    row = {
        'date': int(date_str), 'jcd': jcd, 'rno': rno, 'wind': 0.0,
        'res1': 0, 'rank1': None, 'rank2': None, 'rank3': None,
        'tansho': 0, 'nirentan': 0, 'sanrentan': 0, 'sanrenpuku': 0, 'payout': 0
    }
    # 各艇データ初期化
    for i in range(1, 7):
        row[f'wr{i}'] = 0.0
        row[f'mo{i}'] = 0.0
        row[f'ex{i}'] = 0.0
        row[f'f{i}'] = 0
        row[f'st{i}'] = 0.20

    # --- 2. 天候・風 (BeforeInfo) ---
    try:
        # "Xm" を探すロジック
        wind_txt = ""
        # ラベルから探す
        w_node = soup_before.select_one(".weather1_bodyUnitLabelData")
        if w_node: wind_txt = w_node.text
        else:
            # 見つからない場合は全テキストから検索
            m = re.search(r"風.*?(\d+)m", soup_before.text)
            if m: wind_txt = m.group(1)
        
        m = re.search(r"(\d+)", clean_text(wind_txt))
        if m: row['wind'] = float(m.group(1))
    except: pass

    # --- 3. 各艇データ (BeforeInfo & RaceList) ---
    for i in range(1, 7):
        # 展示タイム (BeforeInfo)
        try:
            cell = soup_before.select_one(f".is-boatColor{i}")
            if cell:
                # ★修正箇所1: tbody -> tr に変更
                # これで行全体ではなく、その艇の行だけを取得します
                tds = cell.find_parent("tr").select("td")
                if len(tds) > 4:
                    val = clean_text(tds[4].text)
                    if re.match(r"\d\.\d{2}", val): row[f'ex{i}'] = float(val)
        except: pass

        # 勝率・モーター・F・ST (RaceList)
        try:
            cell = soup_list.select_one(f".is-boatColor{i}")
            if cell:
                # ★修正箇所2: tbody -> tr に変更
                tds = cell.find_parent("tr").select("td")
                
                # F数 / ST
                if len(tds) > 3:
                    txt = clean_text(tds[3].text)
                    f_m = re.search(r"F(\d+)", txt)
                    if f_m: row[f'f{i}'] = int(f_m.group(1))
                    
                    st_m = re.search(r"(\.\d{2}|\d\.\d{2})", txt)
                    if st_m:
                        v = float(st_m.group(1))
                        if v < 1.0: row[f'st{i}'] = v
                
                # 勝率
                if len(tds) > 4:
                    txt = clean_text(tds[4].text)
                    wr_m = re.search(r"(\d\.\d{2})", txt)
                    if wr_m: row[f'wr{i}'] = float(wr_m.group(1))
                
                # モーター
                if len(tds) > 6:
                    txt = clean_text(tds[6].text)
                    mo_m = re.findall(r"(\d{2,3}\.\d{2})", txt)
                    if mo_m: row[f'mo{i}'] = float(mo_m[0])
        except: pass

    # --- 4. レース結果 (RaceResult) ---
    # まだレースが終わっていない場合は、ここは初期値(None/0)のままになる
    if soup_res:
        try:
            # 順位 (rank1, rank2, rank3)
            # is-w495 テーブルが着順表
            ranks = soup_res.select("table.is-w495 tbody tr")
            if len(ranks) >= 1:
                r1 = clean_text(ranks[0].select("td")[1].text)
                row['rank1'] = int(re.search(r"(\d)", r1).group(1))
            if len(ranks) >= 2:
                r2 = clean_text(ranks[1].select("td")[1].text)
                row['rank2'] = int(re.search(r"(\d)", r2).group(1))
            if len(ranks) >= 3:
                r3 = clean_text(ranks[2].select("td")[1].text)
                row['rank3'] = int(re.search(r"(\d)", r3).group(1))
            
            # res1 (1号艇が1着かどうか)
            if row['rank1'] == 1:
                row['res1'] = 1
            else:
                row['res1'] = 0

            # 払い戻し
            for tbl in soup_res.select("table"):
                txt = clean_text(tbl.text)
                if "勝" in txt or "連" in txt:
                    for tr in tbl.select("tr"):
                        tr_txt = clean_text(tr.text)
                        
                        pay = 0
                        pay_node = tr.select_one(".is-payout1")
                        if pay_node:
                            p_txt = clean_text(pay_node.text).replace("¥","").replace(",","")
                            if p_txt.isdigit(): pay = int(p_txt)

                        if "3連単" in tr_txt:
                            row['sanrentan'] = pay
                            row['payout'] = pay # payoutは3連単配当を入れるのが一般的
                        elif "3連複" in tr_txt:
                            row['sanrenpuku'] = pay
                        elif "2連単" in tr_txt:
                            row['nirentan'] = pay
                        elif "単勝" in tr_txt:
                            row['tansho'] = pay
        except: pass

    # --- 締切時刻 (AI予測に必要なら残す、不要なら削除可) ---
    row['deadline_time'] = "00:00"
    try:
        tgt = soup_list.find(lambda t: "締切" in t.text)
        if tgt:
            tr = tgt.find_parent("tr")
            cells = tr.find_all(['th','td'])
            if len(cells) > rno:
                m = re.search(r"(\d{2}:\d{2})", clean_text(cells[rno].text))
                if m: row['deadline_time'] = m.group(1)
    except: pass

    return row, None

# 互換性のためのダミー
def scrape_result(session, jcd, rno, date_str):
    return None
def scrape_odds(session, jcd, rno, date_str, target_boat=None, target_combo=None):
    return {}
