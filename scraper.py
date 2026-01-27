from curl_cffi import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import re
import unicodedata
import warnings
import time

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

def clean_text(text):
    if not text: return ""
    text = unicodedata.normalize('NFKC', str(text))
    return text.replace("\n", "").replace("\r", "").replace("¥", "").replace(",", "").strip()

def get_session():
    # Chrome 120 偽装 (これでGitHub ActionsのIP規制を突破する)
    return requests.Session(impersonate="chrome120")

def get_soup(session, url):
    try:
        res = session.get(url, timeout=15)
        
        # ★デバッグ用: 失敗時はこれをmain.pyで表示させる
        if res.status_code != 200:
            return None, f"Status {res.status_code}"
            
        # 5KB以下の場合はブロック画面の可能性が高い
        if len(res.content) < 5000:
             # タイトルを取得して確認
             try:
                 chk = BeautifulSoup(res.content, 'lxml')
                 title = chk.title.string if chk.title else "No Title"
             except: title = "Unknown"
             return None, f"BLOCKED? (Size:{len(res.content)}, Title:{title})"
        
        if "データがありません" in res.text:
            return None, "NO_DATA"
            
        return BeautifulSoup(res.content, 'lxml'), None
    except Exception as e:
        return None, f"REQ_ERROR: {e}"

def scrape_race_data(session, jcd, rno, date_str):
    base_url = "https://www.boatrace.jp/owpc/pc/race"
    
    # 3ページ取得
    soup_before, err_b = get_soup(session, f"{base_url}/beforeinfo?rno={rno}&jcd={jcd:02d}&hd={date_str}")
    soup_list, err_l = get_soup(session, f"{base_url}/racelist?rno={rno}&jcd={jcd:02d}&hd={date_str}")
    
    # エラーがあれば理由を返す
    if not soup_before: return None, f"BeforeInfo: {err_b}"
    if not soup_list: return None, f"RaceList: {err_l}"

    row = {'date': date_str, 'jcd': jcd, 'rno': rno}

    # --- ユーザーロジック準拠のデータ抽出 ---
    # 天候・風
    try:
        wind_elem = soup_before.select_one(".weather1_bodyUnitLabelData")
        if wind_elem:
            txt = clean_text(wind_elem.text).replace("m", "").replace(" ", "")
            row['wind'] = float(txt) if txt else 0.0
        else: row['wind'] = 0.0
    except: row['wind'] = 0.0

    # 締切時刻
    row['deadline_time'] = "23:59"
    try:
        target = soup_list.find(lambda t: t.name in ['th','td'] and "締切予定時刻" in t.text)
        if target:
            tr = target.find_parent("tr")
            cells = tr.find_all(['th','td'])
            if len(cells) > rno:
                m = re.search(r"(\d{1,2}:\d{2})", clean_text(cells[rno].text))
                if m: row['deadline_time'] = m.group(1)
    except: pass

    # 各艇データ
    for i in range(1, 7):
        row[f'wr{i}'], row[f'mo{i}'], row[f'ex{i}'] = 0.0, 30.0, 6.80
        row[f'f{i}'], row[f'st{i}'] = 0, 0.20

        # 展示タイム (beforeinfo)
        try:
            boat_cell = soup_before.select_one(f".is-boatColor{i}")
            if boat_cell:
                tds = boat_cell.find_parent("tbody").select("td")
                if len(tds) > 4:
                    ex_val = clean_text(tds[4].text)
                    if re.match(r"\d\.\d{2}", ex_val):
                        row[f'ex{i}'] = float(ex_val)
        except: pass

        # 本番データ (racelist)
        try:
            list_cell = soup_list.select_one(f".is-boatColor{i}")
            if list_cell:
                tds = list_cell.find_parent("tbody").select("td")
                if len(tds) > 3: # ST/F
                    txt = clean_text(tds[3].text)
                    f_match = re.search(r"F(\d+)", txt)
                    if f_match: row[f'f{i}'] = int(f_match.group(1))
                    st_match = re.search(r"(\.\d{2}|\d\.\d{2})", txt)
                    if st_match:
                        val = float(st_match.group(1))
                        if val < 1.0: row[f'st{i}'] = val
                
                if len(tds) > 4: # 勝率
                    txt = tds[4].get_text(" ").strip()
                    wr_match = re.search(r"(\d\.\d{2})", txt)
                    if wr_match: row[f'wr{i}'] = float(wr_match.group(1))

                if len(tds) > 6: # モーター
                    txt = tds[6].get_text(" ").strip()
                    mo_vals = re.findall(r"(\d{1,3}\.\d{2})", txt)
                    if len(mo_vals) >= 1: row[f'mo{i}'] = float(mo_vals[0])
        except: pass

    return row, None

def scrape_result(session, jcd, rno, date_str):
    url = f"https://www.boatrace.jp/owpc/pc/race/raceresult?rno={rno}&jcd={jcd:02d}&hd={date_str}"
    soup, err = get_soup(session, url)
    if not soup: return None
    res = {"nirentan_combo": None, "nirentan_payout": 0}
    try:
        for tbl in soup.select("table"):
            if "2連単" in tbl.text:
                for tr in tbl.select("tr"):
                    if "2連単" in tr.text:
                        nums = tr.select(".numberSet1_number")
                        if len(nums) >= 2:
                            res["nirentan_combo"] = f"{nums[0].text}-{nums[1].text}"
                        pay_node = tr.select_one(".is-payout1")
                        if pay_node:
                            txt = clean_text(pay_node.text)
                            if txt.isdigit():
                                res["nirentan_payout"] = int(txt)
    except: pass
    return res

def scrape_odds(session, jcd, rno, date_str, target_boat=None, target_combo=None):
    return {"tansho": "1.0", "nirentan": "1.0"}
