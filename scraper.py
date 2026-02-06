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
        # 修正: string=re.compile(...) だと空白を含むテキストにヒットしない場合があるため、全探索する
        candidates = soup.find_all(['th', 'td'])
        for tag in candidates:
            if "締切" in tag.text or "予定" in tag.text:
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
                
                # 直後のタグを見る(締切...の次が時間の場合)
                next_tag = tag.find_next_sibling(['td', 'th'])
                if next_tag:
                    text = clean_text(next_tag.text)
                    m = re.search(r"(\d{1,2}:\d{2})", text)
                    if m: return m.group(1).zfill(5)
                
                # 自身のテキストに含まれる場合
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
    soup, status = get_soup(session, url)
    if not soup:
        print(f"⚠️ [3T] スープ取得失敗 {jcd}場{rno}R: {status}")
        return {}

    odds_map = {}
    tables = soup.select("div.table1 table")
    
    for tbl in tables:
        # 修正: "3連単"の文字はテーブル内にはないため、oddsPointクラスの有無で判断する
        if not tbl.select(".oddsPoint"): continue
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
    
    if not odds_map:
        print(f"⚠️ [3T] オッズマップが空です {jcd}場{rno}R (テーブル検出数: {len(tables)})")
        
    return odds_map

def get_odds_2t(session, jcd, rno, date_str):
    url = f"https://www.boatrace.jp/owpc/pc/race/odds2tf?rno={rno}&jcd={jcd:02d}&hd={date_str}"
    soup, status = get_soup(session, url)
    if not soup:
        print(f"⚠️ [2T] スープ取得失敗 {jcd}場{rno}R: {status}")
        return {}
    
    odds_map = {}
    # Use specific class if available, or fallback to all tables (usually .table1 table)
    tables = soup.select("div.table1 table")
    if not tables: tables = soup.select("table")
    
    for tbl in tables:
        # 簡易チェック: 数字アイコンやオッズっぽいセルがあるか
        if not tbl.select(".numberSet1_number") and not tbl.select(".oddsPoint"): 
            continue

        rows = tbl.select("tr")
        
        for tr in rows:
            tds = tr.select("td")
            # 2連単オッズ表は横に6ペア(12セル)並んでいる想定
            if len(tds) < 12: continue 
            
            # 各列が1着艇(1~6)に対応し、セル内が[2着艇, オッズ]
            for i in range(6):
                idx_boat = i * 2
                idx_odd = i * 2 + 1
                if idx_odd >= len(tds): break
                
                try:
                    sec_txt = clean_text(tds[idx_boat].text)
                    odd_txt = clean_text(tds[idx_odd].text)
                    
                    if not sec_txt or not odd_txt: continue
                    
                    sec = int(sec_txt)
                    odd = float(odd_txt)
                    
                    first = i + 1
                    
                    if first != 0 and sec != 0:
                        odds_map[f"{first}-{sec}"] = odd
                except ValueError: 
                    pass
                
    if not odds_map:
        print(f"⚠️ [2T] オッズマップが空です {jcd}場{rno}R (テーブル検出数: {len(tables)})")
        
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
                for tr in rows:
                    if "2連単" in tr.text:
                        combo_node = tr.select(".numberSet1_number")
                        if combo_node:
                            nums = [c.text.strip() for c in combo_node]
                            res['combo_2t'] = "-".join(nums)
                        tds = tr.select("td")
                        for td in reversed(tds):
                            txt = clean_text(td.text).replace("¥","").replace(",","")
                            if txt.isdigit() and int(txt) >= 100:
                                res['payout_2t'] = int(txt); break

    except Exception: pass
    return res
