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
    print(f"DEBUG: Fetching {url_before}")
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
    
    deadline = extract_deadline(soup_before, rno)
    print(f"DEBUG: Extracted deadline: {deadline}")
    row['deadline_time'] = deadline
    
    return row, "OK"

def get_odds_map(session, jcd, rno, date_str):
    url = f"https://www.boatrace.jp/owpc/pc/race/odds3t?rno={rno}&jcd={jcd:02d}&hd={date_str}"
    print(f"DEBUG: Fetching Odds {url}")
    soup, stat = get_soup(session, url)
    if not soup:
        print(f"DEBUG: Odds fetch failed. Stat: {stat}")
        return {}

    with open("debug_odds.html", "w", encoding="utf-8") as f:
        f.write(soup.prettify())
    print("DEBUG: Saved debug_odds.html")

    odds_map = {}
    tables = soup.select("div.table1 table")
    print(f"DEBUG: Found {len(tables)} tables")
    
    for i, tbl in enumerate(tables):
        print(f"DEBUG: Table {i} Text snippet: {repr(tbl.text[:100])}")
        
        # FIX: "3連単" is NOT in the table text, it's in the header. 
        # But the odds table has 'oddsPoint' class in td.
        if not tbl.select(".oddsPoint"):
            print(f"DEBUG: Table {i} skipped (No .oddsPoint class)")
            continue
            
        print(f"DEBUG: Table {i} contains odds data (.oddsPoint found)")

        tbody = tbl.select_one("tbody")
        if not tbody: continue
        rows = tbody.select("tr")
        print(f"DEBUG: Table {i} has {len(rows)} rows")
        
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
    return odds_map

if __name__ == "__main__":
    import datetime
    session = get_session()
    jcd = 11 # Biwako
    rno = 1
    # Use today's date or 20260205
    today = "20260205"
    
    print(f"--- Checking {jcd} R{rno} Date:{today} ---")
    
    # 1. Scrape Info (Deadline)
    row, stat = scrape_race_data(session, jcd, rno, today)
    print(f"Race Info Stat: {stat}")
    print(f"Race Info Row: {row}")
    
    # 2. Get Odds
    odds = get_odds_map(session, jcd, rno, today)
    print(f"Odds Count: {len(odds)}")
    if len(odds) > 0:
        print(f"Sample Odds: {list(odds.items())[:5]}")
    else:
        print("Odds map is empty!")
