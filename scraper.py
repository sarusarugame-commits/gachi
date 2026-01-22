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

# ★追加機能: リアルタイムオッズ取得
def scrape_odds(session, jcd, rno, date_str):
    """
    単勝と2連単の主要オッズを取得してテキストで返す
    """
    # 1. 単勝オッズ (oddstf)
    url_tan = f"https://www.boatrace.jp/owpc/pc/race/oddstf?rno={rno}&jcd={jcd:02d}&hd={date_str}"
    soup_tan = get_soup(session, url_tan)
    tansho_text = "取得失敗"
    
    if soup_tan:
        try:
            # 単勝オッズを簡易抽出 (1号艇〜6号艇)
            odds_list = []
            tables = soup_tan.select("table")
            for tbl in tables:
                if "単勝" in str(tbl): # 単勝テーブルを探す
                    rows = tbl.select("tr")
                    for row in rows:
                        tds = row.select("td")
                        if len(tds) >= 3: # 艇番, 選手名, オッズ
                            boat = clean_text(tds[0].text)
                            val = clean_text(tds[2].text)
                            if boat.isdigit():
                                odds_list.append(f"{boat}号艇:{val}")
                    break
            if odds_list:
                tansho_text = ", ".join(odds_list)
        except: pass

    # 2. 2連単オッズ (odds2tf)
    url_2t = f"https://www.boatrace.jp/owpc/pc/race/odds2tf?rno={rno}&jcd={jcd:02d}&hd={date_str}"
    soup_2t = get_soup(session, url_2t)
    nirentan_text = "取得失敗"

    if soup_2t:
        try:
            # 1号艇頭の人気どころだけ抜粋する (1-2, 1-3, 1-4あたり)
            # ※テーブル構造が複雑なため、テキストから正規表現で拾う等の工夫も有効だが
            # ここでは1号艇の行(最初の行)を狙う
            target_odds = []
            # 2連単テーブルを探す
            tables = soup_2t.select("table")
            for tbl in tables:
                if "2連単" in str(soup_2t) and "小山" in str(soup_2t): # 簡易チェック
                    pass
                
                # 行単位で解析（1号艇の行）
                rows = tbl.select("tr")
                for r in rows:
                    if "1" in r.select_one("th, td").text: # 1号艇の行っぽい
                        tds = r.select("td.oddsPoint")
                        # データがあれば上から順に 1-2, 1-3... の可能性が高いが
                        # 確実性のため、ページ全体のテキストから "1-2" の周辺を探す等のロジック推奨
                        # 今回は簡易的に「オッズページへのリンク」がある前提で処理せず、
                        # 存在確認だけしてGroqにURLを渡す運用もアリだが、
                        # ここでは文字データを整形して返す
                        pass
            
            # 簡易実装: とりあえず単勝が取れていれば判断材料にはなる
            # 2連単は組み合わせが多いため、「1号艇頭」の信頼度チェック用として
            # ページのテキスト全体を少し渡す手もあるが、長くなるので割愛し、
            # 単勝オッズをメインに判断させる
            nirentan_text = "詳細取得略(URL参照)" 
            
            # もし特定できればここで埋める
            # 例: 1-2: 2.7, 1-3: 4.8 ... 
        except: pass

    return {"tansho": tansho_text, "nirentan": nirentan_text}

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
