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

# ★修正: ターゲット（予測した買い目）のオッズをピンポイントで取得
def scrape_odds(session, jcd, rno, date_str, target_boat=None, target_combo=None):
    """
    指定された単勝・2連単オッズを取得する
    target_boat: '1' など
    target_combo: '1-2' など
    """
    result = {"tansho": "---", "nirentan": "---"}

    # 1. 単勝オッズ取得
    if target_boat:
        url_tan = f"https://www.boatrace.jp/owpc/pc/race/oddstf?rno={rno}&jcd={jcd:02d}&hd={date_str}"
        soup_tan = get_soup(session, url_tan)
        if soup_tan:
            try:
                # 艇番を探して、その行のオッズを取得
                # <td class="is-boatColor1">1</td> ... <td class="oddsPoint">1.3</td>
                # クラス名で艇番を特定するのが確実
                boat_class = f"is-boatColor{target_boat}"
                td_boat = soup_tan.find("td", class_=boat_class)
                
                # 見つかったTDの親行(tr)からオッズを探す
                if td_boat:
                    row = td_boat.find_parent("tr")
                    odds_td = row.select_one("td.oddsPoint")
                    if odds_td:
                        result["tansho"] = clean_text(odds_td.text)
            except: pass

    # 2. 2連単オッズ取得
    if target_combo:
        try:
            head, heel = target_combo.split('-') # 1-2 なら head=1, heel=2
            url_2t = f"https://www.boatrace.jp/owpc/pc/race/odds2tf?rno={rno}&jcd={jcd:02d}&hd={date_str}"
            soup_2t = get_soup(session, url_2t)
            
            if soup_2t:
                # 2連単ページは構造が複雑（1号艇頭の表、2号艇頭の表...と並んでいる）
                # head号艇がヘッダーになっているテーブルを探す戦略
                tables = soup_2t.select("div.table1") # 各艇頭のテーブル群
                
                for tbl_div in tables:
                    # そのテーブルが「head号艇」のものか確認
                    # ヘッダーのクラスで判定 (is-boatColor1 など)
                    header_th = tbl_div.select_one(f"th.is-boatColor{head}")
                    if header_th:
                        # このテーブルの中に target_combo があるはず
                        # ヒモ(heel)の番号を持つ td を探す
                        # <td class="is-boatColor2">2</td> ... <td class="oddsPoint">2.7</td>
                        heel_tds = tbl_div.select(f"td.is-boatColor{heel}")
                        for td in heel_tds:
                            # 2連単のヒモ艇番セルか確認（テキストが一致するか）
                            if clean_text(td.text) == str(heel):
                                # その隣(直後)のtdがオッズ
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
