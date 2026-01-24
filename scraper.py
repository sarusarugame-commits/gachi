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
    try:
        target_label = soup.find(lambda tag: tag.name in ['td', 'th'] and "締切予定時刻" in tag.text)
        if target_label:
            parent_row = target_label.find_parent('tr')
            if parent_row:
                cells = parent_row.find_all(['td', 'th'])
                if len(cells) > rno:
                    time_text = clean_text(cells[rno].text)
                    match = re.search(r"(\d{1,2}:\d{2})", time_text)
                    if match: return match.group(1)
    except Exception: pass
    return None

def scrape_odds(session, jcd, rno, date_str, target_boat=None, target_combo=None):
    result = {"tansho": "---", "nirentan": "---"}
    try:
        # 単勝
        if target_boat:
            url_tan = f"https://www.boatrace.jp/owpc/pc/race/oddstf?rno={rno}&jcd={jcd:02d}&hd={date_str}"
            soup_tan = get_soup(session, url_tan)
            if soup_tan:
                td_boat = soup_tan.find("td", class_=f"is-boatColor{target_boat}")
                if td_boat:
                    row = td_boat.find_parent("tr")
                    odds_td = row.select_one("td.oddsPoint")
                    if odds_td: result["tansho"] = clean_text(odds_td.text)

        # 2連単
        if target_combo:
            head, heel = target_combo.split('-')
            url_2t = f"https://www.boatrace.jp/owpc/pc/race/odds2tf?rno={rno}&jcd={jcd:02d}&hd={date_str}"
            soup_2t = get_soup(session, url_2t)
            if soup_2t:
                tables = soup_2t.select("div.table1")
                for tbl in tables:
                    if tbl.select_one(f"th.is-boatColor{head}"):
                        heel_tds = tbl.select(f"td.is-boatColor{heel}")
                        for td in heel_tds:
                            if clean_text(td.text) == str(heel):
                                next_td = td.find_next_sibling("td")
                                if next_td and "oddsPoint" in next_td.get("class", []):
                                    result["nirentan"] = clean_text(next_td.text)
                                    break
    except: pass
    return result

def scrape_race_data(session, jcd, rno, date_str):
    base_url = "https://www.boatrace.jp/owpc/pc/race"
    url_list = f"{base_url}/racelist?rno={rno}&jcd={jcd:02d}&hd={date_str}"
    soup_list = get_soup(session, url_list)
    if not soup_list: return None

    deadline_time = get_deadline_time_accurately(soup_list, rno)
    if not deadline_time: deadline_time = "23:59"

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

# ★修正: 単勝と2連単の結果を正確に取得する
def scrape_result(session, jcd, rno, date_str):
    url = f"https://www.boatrace.jp/owpc/pc/race/raceresult?rno={rno}&jcd={jcd:02d}&hd={date_str}"
    soup = get_soup(session, url)
    
    if not soup: return None
    if "データがありません" in soup.text: return None

    res = {
        "tansho_boat": None, "tansho_payout": 0,
        "nirentan_combo": None, "nirentan_payout": 0
    }

    try:
        # 結果テーブルを探す
        rows = soup.find_all("tr")
        for row in rows:
            th_td = row.find(["th", "td"])
            if not th_td: continue
            
            header_text = clean_text(th_td.text)
            
            # --- 単勝の取得 ---
            if "単勝" in header_text:
                # 艇番: numberSet1_number クラスを持つ span
                boat_span = row.select_one(".numberSet1_number")
                if boat_span:
                    res["tansho_boat"] = clean_text(boat_span.text)
                
                # 払戻金: is-payout1 クラスを持つ span
                payout_span = row.select_one(".is-payout1")
                if payout_span:
                    try:
                        res["tansho_payout"] = int(clean_text(payout_span.text).replace("¥", "").replace(",", ""))
                    except: pass

            # --- 2連単の取得 ---
            elif "2連単" in header_text or "二連単" in header_text:
                # 組番: 複数の numberSet1_number を取得して結合
                boat_spans = row.select(".numberSet1_number")
                if len(boat_spans) >= 2:
                    # [0]が1着, [1]が2着
                    res["nirentan_combo"] = f"{clean_text(boat_spans[0].text)}-{clean_text(boat_spans[1].text)}"
                
                # 払戻金
                payout_span = row.select_one(".is-payout1")
                if payout_span:
                    try:
                        res["nirentan_payout"] = int(clean_text(payout_span.text).replace("¥", "").replace(",", ""))
                    except: pass

        # どちらか片方でも取れていれば成功とみなす
        if res["tansho_boat"] or res["nirentan_combo"]:
            return res

    except Exception as e:
        # print(f"Scrape Error: {e}")
        pass
        
    return None
