import requests
from bs4 import BeautifulSoup
import time
import re
import unicodedata
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

def clean_text(text):
    if not text: return ""
    text = unicodedata.normalize('NFKC', str(text))
    return text.replace("\n", "").replace("\r", "").replace(" ", "").strip()

def extract_float(text):
    if not text: return 0.0
    match = re.search(r"(\d+\.?\d*)", clean_text(text))
    return float(match.group(1)) if match else 0.0

def get_session():
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session

def get_soup(session, url):
    try:
        res = session.get(url, timeout=5) # タイムアウト短縮
        res.encoding = res.apparent_encoding
        return BeautifulSoup(res.text, 'html.parser') if res.status_code == 200 else None
    except: return None

def scrape_race_data(session, jcd, rno, date_str):
    """
    レース情報と【締切時刻】を取得する
    """
    base_url = "https://www.boatrace.jp/owpc/pc/race"
    # 出走表ページ（ここに締切時刻がある）
    url_list = f"{base_url}/racelist?rno={rno}&jcd={jcd:02d}&hd={date_str}"
    soup_list = get_soup(session, url_list)
    if not soup_list: return None

    # 直前情報ページ
    url_before = f"{base_url}/beforeinfo?rno={rno}&jcd={jcd:02d}&hd={date_str}"
    soup_before = get_soup(session, url_before)
    if not soup_before: return None

    row = {'date': date_str, 'jcd': jcd, 'rno': rno}
    
    try:
        # --- 🕒 締切時刻の取得 ---
        # ページ内の "締切予定 10:45" のようなテキストを探す
        body_text = clean_text(soup_list.text)
        time_match = re.search(r"締切予定(\d{1,2}:\d{2})", body_text)
        if time_match:
            row['deadline_time'] = time_match.group(1) # "10:45"
        else:
            row['deadline_time'] = "23:59" # 取得できない場合は夜遅くにしておく

        # --- 以下、データ取得 ---
        weather = soup_before.select(".weather1_bodyUnitLabelData")
        row['wind'] = next((extract_float(e.text) for e in weather if "m" in e.text and "cm" not in e.text), 0.0)
        
        for i in range(1, 7):
            # 展示タイム
            row[f'ex{i}'] = extract_float(soup_before.select_one(f".is-boatColor{i}").find_parent("tbody").select("td")[4].text)
            
            # 本番データ
            tbody = soup_list.select_one(f".is-boatColor{i}").find_parent("tbody")
            tds = tbody.select("td")
            
            row[f'wr{i}'] = extract_float(tds[3].text) # 勝率
            row[f'f{i}'] = int(extract_float(tds[2].text)) # F数
            st_match = re.search(r"ST(\d\.\d{2})", clean_text(tbody.text))
            row[f'st{i}'] = float(st_match.group(1)) if st_match else 0.17 # 平均ST
            row[f'mo{i}'] = extract_float(tds[5].text) or 30.0 # モーター勝率
            
    except: return None
    return row

def scrape_result(session, jcd, rno, date_str):
    url = f"https://www.boatrace.jp/owpc/pc/race/raceresult?rno={rno}&jcd={jcd:02d}&hd={date_str}"
    soup = get_soup(session, url)
    if not soup or "データがありません" in soup.text: return None

    try:
        tables = soup.select(".is-w750 table")
        for table in tables:
            if "二連単" in table.text:
                rows = table.select("tr")
                for r in rows:
                    if "二連単" in r.text:
                        tds = r.select("td")
                        result_combo = clean_text(tds[1].text).replace("-", "-")
                        payout = int(clean_text(tds[2].text).replace("¥", "").replace(",", ""))
                        return {"combo": result_combo, "payout": payout}
    except: pass
    return None
