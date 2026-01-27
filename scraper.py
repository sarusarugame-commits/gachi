import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning # 変更点
import re
import unicodedata
import warnings # 変更点
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

# ⚠️ 警告を無視する設定を追加（これでログが静かになります）
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

def clean_text(text):
    if not text: return ""
    text = unicodedata.normalize('NFKC', str(text))
    return text.replace("\n", " ").replace("\r", "").strip()

def extract_all_numbers(text):
    """テキストから全ての数値を抽出する"""
    if not text: return []
    return re.findall(r"(\d+\.\d+|\d+)", text)

def extract_float(text):
    if not text: return 0.0
    match = re.search(r"(\d+\.?\d*)", clean_text(text))
    return float(match.group(1)) if match else 0.0

def get_session():
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session

def get_soup(session, url):
    try:
        res = session.get(url, timeout=15)
        res.encoding = res.apparent_encoding
        # 警告対策済み
        return BeautifulSoup(res.text, 'lxml') if res.status_code == 200 else None
    except: return None

def scrape_race_data(session, jcd, rno, date_str):
    """AI予測に必要なデータを確実に取得する"""
    base_url = "https://www.boatrace.jp/owpc/pc/race"
    url_before = f"{base_url}/beforeinfo?rno={rno}&jcd={jcd:02d}&hd={date_str}"
    url_list = f"{base_url}/racelist?rno={rno}&jcd={jcd:02d}&hd={date_str}"

    soup_before = get_soup(session, url_before)
    soup_list = get_soup(session, url_list)
    
    if not soup_before or not soup_list: return None
    if "データがありません" in soup_before.text: return None

    row = {'date': date_str, 'jcd': jcd, 'rno': rno}

    # 風速
    try:
        weather = soup_before.select_one(".weather1_bodyUnitLabelData")
        nums = extract_all_numbers(weather.text) if weather else []
        row['wind'] = float(nums[0]) if nums else 0.0
    except: row['wind'] = 0.0

    for i in range(1, 7):
        # --- 展示タイム (beforeinfo) ---
        try:
            boat_node = soup_before.select_one(f"td.is-boatColor{i}")
            ex_val = boat_node.find_parent("tr").select("td")[4].text
            row[f'ex{i}'] = float(re.search(r"(\d\.\d{2})", ex_val).group(1))
        except: row[f'ex{i}'] = 6.80

        # --- 勝率・ST・モーター (racelist) ---
        try:
            list_node = soup_list.select_one(f"td.is-boatColor{i}")
            tbody = list_node.find_parent("tbody")
            tds = tbody.select("td")
            
            wr_cell = tds[3].get_text(separator=' ')
            wr_nums = extract_all_numbers(wr_cell)
            row[f'wr{i}'] = float(wr_nums[0]) if wr_nums else 0.0
            
            all_txt = clean_text(tbody.text)
            st_match = re.search(r"ST(\d\.\d{2})", all_txt.replace(" ", ""))
            row[f'st{i}'] = float(st_match.group(1)) if st_match else 0.17
            
            mo_cell = tds[5].get_text(separator=' ')
            mo_nums = extract_all_numbers(mo_cell)
            row[f'mo{i}'] = float(mo_nums[1]) if len(mo_nums) >= 2 else 30.0
        except:
            row[f'wr{i}'], row[f'st{i}'], row[f'mo{i}'] = 0.0, 0.20, 30.0

    try:
        deadline_time = "23:59"
        target_label = soup_list.find(lambda tag: "締切予定時刻" in tag.text)
        if target_label:
            cells = target_label.find_parent('tr').find_all(['td', 'th'])
            if len(cells) > rno:
                match = re.search(r"(\d{1,2}:\d{2})", cells[rno].text)
                if match: deadline_time = match.group(1)
        row['deadline_time'] = deadline_time
    except: row['deadline_time'] = "23:59"

    return row

def scrape_result(session, jcd, rno, date_str):
    url = f"https://www.boatrace.jp/owpc/pc/race/raceresult?rno={rno}&jcd={jcd:02d}&hd={date_str}"
    soup = get_soup(session, url)
    if not soup or "データがありません" in soup.text: return None
    res = {"nirentan_combo": None, "nirentan_payout": 0}
    try:
        rows = soup.find_all("tr")
        for row in rows:
            txt = clean_text(row.text)
            if "2連単" in txt or "二連単" in txt:
                nums = row.select(".numberSet1_number")
                if len(nums) >= 2:
                    res["nirentan_combo"] = f"{nums[0].text}-{nums[1].text}"
                pay = row.select_one(".is-payout1")
                if pay: res["nirentan_payout"] = int(pay.text.replace("¥","").replace(",",""))
        return res
    except: return None

def scrape_odds(session, jcd, rno, date_str, target_boat=None, target_combo=None):
    res_odds = {"tansho": "1.0", "nirentan": "1.0"}
    try:
        if target_combo:
            head, heel = target_combo.split('-')
            url = f"https://www.boatrace.jp/owpc/pc/race/odds2tf?rno={rno}&jcd={jcd:02d}&hd={date_str}"
            soup = get_soup(session, url)
            if soup:
                for tbl in soup.select("div.table1"):
                    if tbl.select_one(f"th.is-boatColor{head}"):
                        for td in tbl.select(f"td.is-boatColor{heel}"):
                            if clean_text(td.text) == str(heel):
                                val = td.find_next_sibling("td").text
                                res_odds["nirentan"] = val.strip()
    except: pass
    return res_odds
