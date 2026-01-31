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
    # Chrome 120 偽装
    return requests.Session(impersonate="chrome120")

def get_soup(session, url):
    try:
        res = session.get(url, timeout=10)
        if res.status_code != 200: return None
        if len(res.content) < 3000: return None 
        if "データがありません" in res.text: return None
        return BeautifulSoup(res.content, 'lxml')
    except: return None

def extract_deadline(soup, rno):
    """HTML構造解析による強力な締切時刻取得"""
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
    soup_before = get_soup(session, url_before)
    
    soup_list = None
    url_list = f"{base_url}/racelist?rno={rno}&jcd={jcd:02d}&hd={date_str}"
    soup_list = get_soup(session, url_list)

    if not soup_before and not soup_list:
        return None, "NO_DATA"

    row = {
        'date': int(date_str), 'jcd': jcd, 'rno': rno, 'wind': 0.0,
        'deadline_time': None,
        'pid1':0, 'wr1':0.0, 'mo1':0.0, 'ex1':0.0, 'f1':0, 'st1':0.20,
        'pid2':0, 'wr2':0.0, 'mo2':0.0, 'ex2':0.0, 'f2':0, 'st2':0.20,
        'pid3':0, 'wr3':0.0, 'mo3':0.0, 'ex3':0.0, 'f3':0, 'st3':0.20,
        'pid4':0, 'wr4':0.0, 'mo4':0.0, 'ex4':0.0, 'f4':0, 'st4':0.20,
        'pid5':0, 'wr5':0.0, 'mo5':0.0, 'ex5':0.0, 'f5':0, 'st5':0.20,
        'pid6':0, 'wr6':0.0, 'mo6':0.0, 'ex6':0.0, 'f6':0, 'st6':0.20,
    }

    # 締切時刻
    row['deadline_time'] = extract_deadline(soup_before, rno)
    if not row['deadline_time']:
        row['deadline_time'] = extract_deadline(soup_list, rno)

    # 天候・風
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

    # --- 各艇データ ---
    for i in range(1, 7):
        # ★重要: 展示タイムは直前情報からのみ取得（出走表の勝率と混同しない）
        if soup_before:
            try:
                boat_td = soup_before.select_one(f"td.is-boatColor{i}")
                if boat_td:
                    tr = boat_td.find_parent("tr")
                    if tr:
                        text_all = clean_text(tr.text)
                        # 6.xx または 7.00~7.49 あたりを展示タイムとみなす
                        matches = re.findall(r"(6\.\d{2}|7\.[0-4]\d)", text_all)
                        if matches:
                            row[f'ex{i}'] = float(matches[-1])
            except: pass

        # 出走表データ（勝率・モーターなど）
        if soup_list:
            try:
                tbodies = soup_list.select("tbody.is-fs12")
                if len(tbodies) >= i:
                    tbody = tbodies[i-1]
                    txt_all = clean_text(tbody.text)
                    
                    pid_match = re.search(r"([2-5]\d{3})", txt_all)
                    if pid_match: row[f'pid{i}'] = int(pid_match.group(1))

                    full_row_text = txt_all 

                    # 勝率
                    wr_matches = re.findall(r"(\d\.\d{2})", full_row_text)
                    for val_str in wr_matches:
                        val = float(val_str)
                        if 1.0 <= val <= 9.99:
                            row[f'wr{i}'] = val
                            break
                    
                    # モーター
                    mo_matches = re.findall(r"(\d{2}\.\d{2})", full_row_text)
                    for m_val in mo_matches:
                        if 10.0 <= float(m_val) <= 99.9:
                            row[f'mo{i}'] = float(m_val)
                            break
                    
                    # 平均ST
                    st_match = re.search(r"(0\.\d{2})", full_row_text)
                    if st_match: row[f'st{i}'] = float(st_match.group(1))

                    # F数
                    f_match = re.search(r"F(\d+)", full_row_text)
                    if f_match: row[f'f{i}'] = int(f_match.group(1))

            except: pass

    return row, None

def get_exact_odds(session, jcd, rno, date_str, combo):
    """指定された買い目（例: '1-2-3'）の現在のオッズを取得する"""
    url = f"https://www.boatrace.jp/owpc/pc/race/odds3t?rno={rno}&jcd={jcd:02d}&hd={date_str}"
    soup = get_soup(session, url)
    if not soup: return None

    try:
        # 買い目を含むtdを探す
        target_td = soup.find(lambda tag: tag.name == "td" and combo in clean_text(tag.text))
        if target_td:
            odds_td = target_td.find_next_sibling("td")
            if odds_td:
                odds_txt = clean_text(odds_td.text)
                m = re.search(r"(\d{1,3}\.\d)", odds_txt)
                if m:
                    return float(m.group(1))
    except Exception:
        pass
    return None

def scrape_result(session, jcd, rno, date_str):
    url = f"https://www.boatrace.jp/owpc/pc/race/raceresult?rno={rno}&jcd={jcd:02d}&hd={date_str}"
    soup = get_soup(session, url)
    if not soup: return None

    res = { 'sanrentan_combo': None, 'sanrentan_payout': 0 }
    try:
        tables = soup.select("table.is-w495")
        for tbl in tables:
            if "3連単" in tbl.text:
                rows = tbl.select("tr")
                for tr in rows:
                    if "3連単" in tr.text:
                        combo_node = tr.select(".numberSet1_number")
                        if combo_node:
                            nums = [c.text.strip() for c in combo_node]
                            res['sanrentan_combo'] = "-".join(nums)
                        
                        tds = tr.select("td")
                        for td in reversed(tds):
                            txt = clean_text(td.text).replace("¥","").replace(",","")
                            if txt.isdigit():
                                val = int(txt)
                                if val >= 100:
                                    res['sanrentan_payout'] = val
                                    break
    except Exception: pass
    if not res['sanrentan_combo']: return None
    return res

def scrape_odds(session, jcd, rno, date_str):
    return {}
