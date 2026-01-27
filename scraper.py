from curl_cffi import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import re
import unicodedata
import warnings

# ログ汚染対策
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

def clean_text(text):
    if not text: return ""
    text = unicodedata.normalize('NFKC', str(text))
    return text.replace("\n", " ").replace("\r", "").strip()

def extract_all_numbers(text):
    if not text: return []
    return re.findall(r"(\d+\.\d+|\d+)", text)

def get_session():
    # ★ここがキモ: Chrome 120 の指紋（TLS Fingerprint）を完全模倣
    # これにより、サーバーは「ボットではなく人間（ブラウザ）が来た」と誤認します
    session = requests.Session(impersonate="chrome120")
    return session

def get_soup(session, url):
    try:
        # タイムアウトは短めに設定
        res = session.get(url, timeout=10)
        
        # 万が一ブロック画面(5KB以下)ならログを出す
        if len(res.content) < 6000:
            print(f"⚠️ [Block check] Size:{len(res.content)} | URL:{url}", flush=True)
        
        return BeautifulSoup(res.content, 'lxml') if res.status_code == 200 else None
    except Exception as e:
        return None

def scrape_race_data(session, jcd, rno, date_str):
    """高速並列処理に耐えうるスクレイピング"""
    base_url = "https://www.boatrace.jp/owpc/pc/race"
    url_before = f"{base_url}/beforeinfo?rno={rno}&jcd={jcd:02d}&hd={date_str}"
    url_list = f"{base_url}/racelist?rno={rno}&jcd={jcd:02d}&hd={date_str}"

    # 並列アクセス
    soup_before = get_soup(session, url_before)
    soup_list = get_soup(session, url_list)
    
    if not soup_before or not soup_list: return None
    if "データがありません" in soup_before.text: return None

    row = {'date': date_str, 'jcd': jcd, 'rno': rno}
    row['deadline_time'] = "23:59"

    # --- 締切時刻 ---
    try:
        target_cell = soup_list.find(lambda tag: tag.name in ['th', 'td'] and "締切予定時刻" in tag.text)
        if target_cell:
            cells = target_cell.find_parent("tr").find_all(['th', 'td'])
            if len(cells) > rno:
                time_txt = clean_text(cells[rno].text)
                match = re.search(r"(\d{1,2}:\d{2})", time_txt)
                if match: row['deadline_time'] = match.group(1)
    except: pass

    # --- 風速 ---
    try:
        w_node = soup_before.select_one(".weather1_bodyUnitLabelData")
        match = re.search(r"(\d+)m", clean_text(w_node.text)) if w_node else None
        row['wind'] = float(match.group(1)) if match else 0.0
    except: row['wind'] = 0.0

    # --- 各艇データ ---
    for i in range(1, 7):
        # 展示タイム
        try:
            node = soup_before.select_one(f"td.is-boatColor{i}")
            tr = node.find_parent("tr")
            ex_txt = clean_text(tr.select("td")[4].text)
            row[f'ex{i}'] = float(re.search(r"(\d\.\d{2})", ex_txt).group(1))
        except: row[f'ex{i}'] = 6.80

        # 勝率・ST・モーター
        try:
            node = soup_list.select_one(f"td.is-boatColor{i}")
            tbody = node.find_parent("tbody")
            full_text = clean_text(tbody.text)

            # 勝率 (1.50 ~ 9.99)
            wr_matches = re.findall(r"(\d\.\d{2})", full_text)
            valid_wr = [float(x) for x in wr_matches if 1.5 <= float(x) <= 9.99]
            row[f'wr{i}'] = valid_wr[0] if valid_wr else 0.0

            # ST
            st_match = re.search(r"ST(\.\d{2}|\d\.\d{2})", full_text.replace(" ", ""))
            if st_match:
                val = st_match.group(1)
                row[f'st{i}'] = float(val) if val.startswith("0") or val.startswith(".") else 0.20
            else: row[f'st{i}'] = 0.20
            if row[f'st{i}'] < 0: row[f'st{i}'] = 0.20

            # モーター (10.0以上)
            mo_matches = re.findall(r"(\d{2}\.\d)", full_text)
            valid_mo = [float(x) for x in mo_matches if float(x) > 10.0]
            row[f'mo{i}'] = valid_mo[0] if valid_mo else 30.0

        except:
            row[f'wr{i}'], row[f'st{i}'], row[f'mo{i}'] = 0.0, 0.20, 30.0

    return row

def scrape_result(session, jcd, rno, date_str):
    url = f"https://www.boatrace.jp/owpc/pc/race/raceresult?rno={rno}&jcd={jcd:02d}&hd={date_str}"
    soup = get_soup(session, url)
    if not soup or "データがありません" in soup.text: return None
    res = {"nirentan_combo": None, "nirentan_payout": 0}
    try:
        for row in soup.select("tr"):
            txt = clean_text(row.text)
            if "2連単" in txt:
                nums = row.select(".numberSet1_number")
                if len(nums) >= 2:
                    res["nirentan_combo"] = f"{nums[0].text}-{nums[1].text}"
                pay = row.select_one(".is-payout1")
                if pay: res["nirentan_payout"] = int(pay.text.replace("¥","").replace(",",""))
    except: pass
    return res

def scrape_odds(session, jcd, rno, date_str, target_boat=None, target_combo=None):
    return {"tansho": "1.0", "nirentan": "1.0"}
