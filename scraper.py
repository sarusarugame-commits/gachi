# scraper.py
from curl_cffi import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import re
import unicodedata
import warnings

# ... (clean_text, get_session はそのまま) ...

def get_soup(session, url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://www.boatrace.jp/",
            "Accept-Language": "ja,en-US;q=0.9,en;q=0.8"
        }
        res = session.get(url, headers=headers, timeout=15)
        
        # 明確な「データなし」判定
        if "データがありません" in res.text: return None, "NO_RACE"
        if res.status_code == 404: return None, "NO_RACE"
        
        # その他のエラー（リトライ対象）
        if res.status_code != 200: return None, "HTTP_ERROR"
        if len(res.content) < 1000: return None, "SMALL_CONTENT"
        
        return BeautifulSoup(res.content, 'lxml'), "OK"
    except Exception as e: return None, f"EXCEPTION_{e}"

# ... (extract_deadline はそのまま) ...

def scrape_race_data(session, jcd, rno, date_str):
    base_url = "https://www.boatrace.jp/owpc/pc/race"
    url_before = f"{base_url}/beforeinfo?rno={rno}&jcd={jcd:02d}&hd={date_str}"
    # 戻り値を2つ受け取るように変更
    soup_before, stat_b = get_soup(session, url_before)
    
    url_list = f"{base_url}/racelist?rno={rno}&jcd={jcd:02d}&hd={date_str}"
    soup_list, stat_l = get_soup(session, url_list)

    # どちらかが明確に「開催なし」なら、そのレースは存在しないとみなす
    if stat_b == "NO_RACE" or stat_l == "NO_RACE":
        return None, "NO_RACE"

    # 両方とも取得失敗（通信エラーなど）の場合はリトライ対象のエラー
    if not soup_before and not soup_list: 
        return None, f"FETCH_ERR({stat_b}/{stat_l})"

    row = {
        'date': int(date_str), 'jcd': jcd, 'rno': rno, 'wind': 0.0,
        'deadline_time': None
    }
    
    # ... (以降のデータ抽出ロジックは元のまま変更なし) ...
    # ※ただし extract_deadline 等の呼び出しで soup_before が None の場合のガードは既存コードに入っているのでそのままでOK
    
    # 特徴量の初期化
    for i in range(1, 7):
        row[f'pid{i}'] = 0
        row[f'wr{i}'] = 0.0
        row[f'mo{i}'] = 0.0
        row[f'ex{i}'] = 0.0
        row[f'f{i}'] = 0
        row[f'st{i}'] = 0.20

    # 締切時間取得
    row['deadline_time'] = extract_deadline(soup_before, rno)
    if not row['deadline_time']:
        row['deadline_time'] = extract_deadline(soup_list, rno)
        
    # 風速取得
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

    # 各艇データの取得
    for i in range(1, 7):
        # 直前情報から展示タイム
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
            
        # 出走表から選手スペック
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
            
    return row, None
