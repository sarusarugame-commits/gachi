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
        if len(res.content) < 3000: return None # Block check
        if "データがありません" in res.text: return None
        return BeautifulSoup(res.content, 'lxml')
    except: return None

def scrape_race_data(session, jcd, rno, date_str):
    """
    予想に必要なデータ（出走表、直前情報）を取得する
    """
    base_url = "https://www.boatrace.jp/owpc/pc/race"
    
    soup_before = get_soup(session, f"{base_url}/beforeinfo?rno={rno}&jcd={jcd:02d}&hd={date_str}")
    soup_list = get_soup(session, f"{base_url}/racelist?rno={rno}&jcd={jcd:02d}&hd={date_str}")

    if not soup_before or not soup_list:
        return None, "NO_DATA"

    row = {
        'date': int(date_str), 'jcd': jcd, 'rno': rno, 'wind': 0.0,
        # 各艇データ初期化
        'pid1':0, 'wr1':0.0, 'mo1':0.0, 'ex1':0.0, 'f1':0, 'st1':0.20,
        'pid2':0, 'wr2':0.0, 'mo2':0.0, 'ex2':0.0, 'f2':0, 'st2':0.20,
        'pid3':0, 'wr3':0.0, 'mo3':0.0, 'ex3':0.0, 'f3':0, 'st3':0.20,
        'pid4':0, 'wr4':0.0, 'mo4':0.0, 'ex4':0.0, 'f4':0, 'st4':0.20,
        'pid5':0, 'wr5':0.0, 'mo5':0.0, 'ex5':0.0, 'f5':0, 'st5':0.20,
        'pid6':0, 'wr6':0.0, 'mo6':0.0, 'ex6':0.0, 'f6':0, 'st6':0.20,
    }

    # --- 天候・風 ---
    try:
        w_node = soup_before.select_one(".weather1_bodyUnitLabelData")
        wind_txt = w_node.text if w_node else ""
        if not wind_txt:
            m = re.search(r"風.*?(\d+)m", soup_before.text)
            if m: wind_txt = m.group(1)
        m = re.search(r"(\d+)", clean_text(wind_txt))
        if m: row['wind'] = float(m.group(1))
    except: pass

    # --- 各艇データ ---
    for i in range(1, 7):
        # 展示タイム
        try:
            cell = soup_before.select_one(f".is-boatColor{i}")
            if cell:
                tds = cell.find_parent("tr").select("td")
                if len(tds) > 4:
                    val = clean_text(tds[4].text)
                    if re.match(r"\d\.\d{2}", val): row[f'ex{i}'] = float(val)
        except: pass

        # 出走表データ
        try:
            cell = soup_list.select_one(f".is-boatColor{i}")
            if cell:
                tbody = cell.find_parent("tbody")
                # PID
                pid_node = tbody.select_one(".is-fs11")
                if pid_node:
                    pm = re.search(r"(\d{4})", pid_node.text)
                    if pm: row[f'pid{i}'] = int(pm.group(1))

                tr = cell.find_parent("tr")
                tds = tr.select("td")
                
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

    return row, None


def scrape_result(session, jcd, rno, date_str):
    """
    結果ページを解析し、決まり手と配当を返す (main.pyのreport_worker用)
    """
    url = f"https://www.boatrace.jp/owpc/pc/race/raceresult?rno={rno}&jcd={jcd:02d}&hd={date_str}"
    soup = get_soup(session, url)
    if not soup: return None

    res = {
        'sanrentan_combo': None,
        'sanrentan_payout': 0,
        'nirentan_combo': None,
        'nirentan_payout': 0,
        'tansho_boat': None,
        'tansho_payout': 0
    }

    try:
        # 配当テーブルを探す
        # 通常、クラス名 is-w495 などがついているが、構造から探す
        tables = soup.select("table")
        for tbl in tables:
            txt = clean_text(tbl.text)
            if "3連単" in txt and "払戻金" in txt:
                rows = tbl.select("tr")
                for tr in rows:
                    row_txt = clean_text(tr.text)
                    
                    # 組み合わせ (例: 1-2-3)
                    # td class="is-payout2" にあることが多いが、構造が変わることもあるので
                    # "numberSet1_number" クラスを探す
                    combo_node = tr.select(".numberSet1_number")
                    combo_text = ""
                    if combo_node:
                        # 1, 2, 3 のように分かれているので結合
                        nums = [c.text.strip() for c in combo_node]
                        combo_text = "-".join(nums)
                    
                    # 配当金
                    pay_node = tr.select_one(".is-payout1")
                    payout = 0
                    if pay_node:
                        p_txt = clean_text(pay_node.text).replace("¥","").replace(",","")
                        if p_txt.isdigit(): payout = int(p_txt)

                    if "3連単" in row_txt and combo_text:
                        res['sanrentan_combo'] = combo_text
                        res['sanrentan_payout'] = payout
                    elif "2連単" in row_txt and combo_text:
                        res['nirentan_combo'] = combo_text
                        res['nirentan_payout'] = payout
                    elif "単勝" in row_txt and combo_text:
                        res['tansho_boat'] = combo_text # 単勝は "1" などの単一数字
                        res['tansho_payout'] = payout

    except Exception:
        pass

    # 結果が取れていなければNone (まだレース終わってない等)
    if not res['sanrentan_combo']:
        return None
        
    return res

def scrape_odds(session, jcd, rno, date_str):
    # 今回は使用しないので空実装でOK
    return {}
