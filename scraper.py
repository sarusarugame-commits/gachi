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

def get_deadline_time_accurately(soup, rno):
    """
    HTML構造から、対象レース(rno)の締切時刻をピンポイントで抜く
    """
    try:
        # 1. "締切予定時刻" という文字を含むセルを探す
        # HTML解析: <td ...>締切予定時刻</td> というタグが存在する
        target_label = soup.find(lambda tag: tag.name in ['td', 'th'] and "締切予定時刻" in tag.text)
        
        if target_label:
            # 2. その親の行(tr)を取得
            parent_row = target_label.find_parent('tr')
            
            if parent_row:
                # 3. その行の中にある全てのセル(td/th)を取得
                cells = parent_row.find_all(['td', 'th'])
                
                # 解説: 
                # cells[0] は「締切予定時刻」という見出しセル
                # cells[1] は 1Rの時刻
                # cells[2] は 2Rの時刻
                # ...
                # cells[rno] が そのレースの時刻 になる
                
                if len(cells) > rno:
                    time_text = clean_text(cells[rno].text)
                    match = re.search(r"(\d{1,2}:\d{2})", time_text)
                    if match:
                        return match.group(1)

    except Exception:
        pass
    
    return None

def scrape_race_data(session, jcd, rno, date_str):
    base_url = "https://www.boatrace.jp/owpc/pc/race"
    url_list = f"{base_url}/racelist?rno={rno}&jcd={jcd:02d}&hd={date_str}"
    soup_list = get_soup(session, url_list)
    if not soup_list: return None

    # ★修正点: rno を渡して、そのレース番号の時刻を取得する
    deadline_time = get_deadline_time_accurately(soup_list, rno)
    
    # 取得できない場合はNone (main.py側で処理)
    if not deadline_time:
        deadline_time = "23:59"

    url_before = f"{base_url}/beforeinfo?rno={rno}&jcd={jcd:02d}&hd={date_str}"
    soup_before = get_soup(session, url_before)
    if not soup_before: return None

    row = {'date': date_str, 'jcd': jcd, 'rno': rno, 'deadline_time': deadline_time}
    
    try:
        weather = soup_before.select(".weather1_bodyUnitLabelData")
        row['wind'] = next((extract_float(e.text) for e in weather if "m" in e.text and "cm" not in e.text), 0.0)
        
        for i in range(1, 7):
            try:
                node = soup_before.select_one(f".is-boatColor{i}")
                val = node.find_parent("tbody").select("td")[4].text if node else "6.80"
                row[f'ex{i}'] = extract_float(val)
            except: row[f'ex{i}'] = 6.80

            try:
                node_list = soup_list.select_one(f".is-boatColor{i}")
                if not node_list: return None
                tbody = node_list.find_parent("tbody")
                tds = tbody.select("td")
                
                row[f'wr{i}'] = extract_float(tds[3].text)
                row[f'f{i}'] = int(extract_float(tds[2].text))
                st_match = re.search(r"ST(\d\.\d{2})", clean_text(tbody.text))
                row[f'st{i}'] = float(st_match.group(1)) if st_match else 0.17
                row[f'mo{i}'] = extract_float(tds[5].text) or 30.0
            except: return None
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
