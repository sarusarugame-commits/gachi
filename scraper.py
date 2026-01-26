import requests
from bs4 import BeautifulSoup
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
        res = session.get(url, timeout=10)
        res.encoding = res.apparent_encoding
        return BeautifulSoup(res.text, 'html.parser') if res.status_code == 200 else None
    except: return None

def scrape_race_data(session, jcd, rno, date_str):
    base_url = "https://www.boatrace.jp/owpc/pc/race"
    
    # 1. 直前情報 (風、展示タイム)
    url_before = f"{base_url}/beforeinfo?rno={rno}&jcd={jcd:02d}&hd={date_str}"
    soup_before = get_soup(session, url_before)
    if not soup_before: return None
    
    if "データがありません" in soup_before.text: return None

    # 2. 番組表 (勝率、モーター、ST)
    url_list = f"{base_url}/racelist?rno={rno}&jcd={jcd:02d}&hd={date_str}"
    soup_list = get_soup(session, url_list)
    if not soup_list: return None

    # 締切時刻取得
    deadline_time = "23:59"
    try:
        target_label = soup_list.find(lambda tag: tag.name in ['td', 'th'] and "締切予定時刻" in tag.text)
        if target_label:
            parent_row = target_label.find_parent('tr')
            if parent_row:
                cells = parent_row.find_all(['td', 'th'])
                if len(cells) > rno:
                    match = re.search(r"(\d{1,2}:\d{2})", clean_text(cells[rno].text))
                    if match: deadline_time = match.group(1)
    except: pass

    row = {'date': date_str, 'jcd': jcd, 'rno': rno, 'deadline_time': deadline_time}
    
    # 風速
    try:
        weather = soup_before.select(".weather1_bodyUnitLabelData")
        row['wind'] = next((extract_float(e.text) for e in weather if "m" in e.text and "cm" not in e.text), 0.0)
    except: row['wind'] = 0.0
        
    for i in range(1, 7):
        # 展示タイム
        try:
            node = soup_before.select_one(f".is-boatColor{i}")
            val = node.find_parent("tbody").select("td")[4].text if node else "6.80"
            row[f'ex{i}'] = extract_float(val)
        except: row[f'ex{i}'] = 6.80

        # 勝率・モーター・ST
        try:
            node_list = soup_list.select_one(f".is-boatColor{i}")
            if not node_list: return None
            tbody = node_list.find_parent("tbody")
            tds = tbody.select("td")
            
            row[f'wr{i}'] = extract_float(tds[3].text)
            st_match = re.search(r"ST(\d\.\d{2})", clean_text(tbody.text))
            row[f'st{i}'] = float(st_match.group(1)) if st_match else 0.20
            row[f'mo{i}'] = extract_float(tds[5].text) or 30.0
        except: return None

    return row

def scrape_result(session, jcd, rno, date_str):
    """レース結果取得（変更なし）"""
    url = f"https://www.boatrace.jp/owpc/pc/race/raceresult?rno={rno}&jcd={jcd:02d}&hd={date_str}"
    soup = get_soup(session, url)
    if not soup or "データがありません" in soup.text: return None

    res = {"tansho_boat": None, "tansho_payout": 0, "nirentan_combo": None, "nirentan_payout": 0}
    try:
        rows = soup.find_all("tr")
        for row in rows:
            th_td = row.find(["th", "td"])
            if not th_td: continue
            header_text = clean_text(th_td.text)
            
            if "単勝" in header_text:
                boat = row.select_one(".numberSet1_number")
                pay = row.select_one(".is-payout1")
                if boat: res["tansho_boat"] = clean_text(boat.text)
                if pay: res["tansho_payout"] = int(clean_text(pay.text).replace("¥","").replace(",",""))
            
            elif "2連単" in header_text or "二連単" in header_text:
                boats = row.select(".numberSet1_number")
                pay = row.select_one(".is-payout1")
                if len(boats) >= 2: 
                    res["nirentan_combo"] = f"{clean_text(boats[0].text)}-{clean_text(boats[1].text)}"
                if pay: res["nirentan_payout"] = int(clean_text(pay.text).replace("¥","").replace(",",""))
        
        if res["tansho_boat"] or res["nirentan_combo"]: return res
    except: pass
    return None

def scrape_odds(session, jcd, rno, date_str, target_boat=None, target_combo=None):
    """オッズ取得（簡易版）"""
    result = {"tansho": "1.0", "nirentan": "1.0"}
    try:
        # 単勝
        if target_boat:
            url = f"https://www.boatrace.jp/owpc/pc/race/oddstf?rno={rno}&jcd={jcd:02d}&hd={date_str}"
            soup = get_soup(session, url)
            if soup:
                td = soup.find("td", class_=f"is-boatColor{target_boat}")
                if td:
                    odds = td.find_parent("tr").select_one("td.oddsPoint")
                    if odds: result["tansho"] = clean_text(odds.text)
        # 2連単
        if target_combo:
            head, heel = target_combo.split('-')
            url = f"https://www.boatrace.jp/owpc/pc/race/odds2tf?rno={rno}&jcd={jcd:02d}&hd={date_str}"
            soup = get_soup(session, url)
            if soup:
                for tbl in soup.select("div.table1"):
                    if tbl.select_one(f"th.is-boatColor{head}"):
                        for td in tbl.select(f"td.is-boatColor{heel}"):
                            if clean_text(td.text) == str(heel):
                                nxt = td.find_next_sibling("td")
                                if nxt: result["nirentan"] = clean_text(nxt.text)
    except: pass
    return result
