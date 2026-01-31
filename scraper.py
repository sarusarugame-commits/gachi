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
        # ヘッダー強化
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://www.boatrace.jp/"
        }
        res = session.get(url, headers=headers, timeout=15)
        if res.status_code != 200: return None
        if len(res.content) < 1000: return None 
        if "データがありません" in res.text: return None
        return BeautifulSoup(res.content, 'lxml')
    except: return None

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
    soup_before = get_soup(session, url_before)
    soup_list = None
    url_list = f"{base_url}/racelist?rno={rno}&jcd={jcd:02d}&hd={date_str}"
    soup_list = get_soup(session, url_list)

    if not soup_before and not soup_list: return None, "NO_DATA"

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
        # 展示タイムは直前情報からのみ
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
        
        # 出走表情報
        if soup_list:
            try:
                tbodies = soup_list.select("tbody.is-fs12")
                if len(tbodies) >= i:
                    tbody = tbodies[i-1]
                    txt_all = clean_text(tbody.text)
                    pid_match = re.search(r"([2-5]\d{3})", txt_all)
                    if pid_match: row[f'pid{i}'] = int(pid_match.group(1))
                    full_row_text = txt_all 
                    wr_matches = re.findall(r"(\d\.\d{2})", full_row_text)
                    for val_str in wr_matches:
                        val = float(val_str)
                        if 1.0 <= val <= 9.99:
                            row[f'wr{i}'] = val
                            break
                    mo_matches = re.findall(r"(\d{2}\.\d{2})", full_row_text)
                    for m_val in mo_matches:
                        if 10.0 <= float(m_val) <= 99.9:
                            row[f'mo{i}'] = float(m_val)
                            break
                    st_match = re.search(r"(0\.\d{2})", full_row_text)
                    if st_match: row[f'st{i}'] = float(st_match.group(1))
                    f_match = re.search(r"F(\d+)", full_row_text)
                    if f_match: row[f'f{i}'] = int(f_match.group(1))
            except: pass
    return row, None

def get_exact_odds(session, jcd, rno, date_str, combo):
    """
    rowspan（結合セル）で崩れたテーブルから正確にオッズを抜き出すロジック
    """
    url = f"https://www.boatrace.jp/owpc/pc/race/odds3t?rno={rno}&jcd={jcd:02d}&hd={date_str}"
    soup = get_soup(session, url)
    if not soup: return None

    try:
        t_b1, t_b2, t_b3 = map(int, combo.split('-'))
    except: return None

    tables = soup.select("div.table1 table")
    target_tbody = None
    for tbl in tables:
        if "3連単" in tbl.text or "締切" not in tbl.text:
            tbody = tbl.select_one("tbody")
            if tbody:
                target_tbody = tbody
                break
    
    if not target_tbody: return None
    rows = target_tbody.select("tr")

    rowspan_counters = [0] * 6
    current_2nd_boats = [0] * 6

    for r_idx, tr in enumerate(rows):
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
                col_cursor += 2
                rowspan_counters[block_idx] -= 1
            else:
                if col_cursor + 2 >= len(tds): break
                td_2nd = tds[col_cursor]
                txt_2nd = clean_text(td_2nd.text)
                rs = int(td_2nd.get("rowspan", 1))
                rowspan_counters[block_idx] = rs - 1
                try:
                    val_2nd = int(txt_2nd)
                    current_2nd_boats[block_idx] = val_2nd
                except: val_2nd = 0
                txt_3rd = clean_text(tds[col_cursor+1].text)
                txt_odds = clean_text(tds[col_cursor+2].text)
                col_cursor += 3

            try:
                val_3rd = int(txt_3rd)
                if current_1st == t_b1 and val_2nd == t_b2 and val_3rd == t_b3:
                    return float(txt_odds)
            except: continue

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
                                if val >= 100: res['sanrentan_payout'] = val; break
    except Exception: pass
    if not res['sanrentan_combo']: return None
    return res

def scrape_odds(session, jcd, rno, date_str):
    return {}
