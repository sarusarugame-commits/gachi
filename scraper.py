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
    return requests.Session(impersonate="chrome120")

def get_soup(session, url):
    try:
        res = session.get(url, timeout=10)
        if res.status_code != 200: return None
        if len(res.content) < 3000: return None 
        if "データがありません" in res.text: return None
        return BeautifulSoup(res.content, 'lxml')
    except: return None

# ... (extract_deadline, scrape_race_data, scrape_result は変更なし。そのまま維持してください) ...
# ★以前のコードの scrape_race_data 等はそのまま残してください

def get_exact_odds(session, jcd, rno, date_str, combo):
    """
    指定された買い目（例: '1-2-3'）の現在のオッズを取得する
    """
    url = f"https://www.boatrace.jp/owpc/pc/race/odds3t?rno={rno}&jcd={jcd:02d}&hd={date_str}"
    soup = get_soup(session, url)
    if not soup: return None

    try:
        # 3連単オッズページから買い目を探す
        # 買い目は "1-2-3" のようなテキストでtdに入っている
        
        # 1. 買い目そのものを探す
        # ページ内のすべてのセルから完全一致を探すのが最も確実
        target_td = soup.find(lambda tag: tag.name == "td" and combo in clean_text(tag.text))
        
        if target_td:
            # オッズは通常、その次のtdにある
            odds_td = target_td.find_next_sibling("td")
            if odds_td:
                odds_txt = clean_text(odds_td.text)
                # "12.5" のような数値を抽出
                m = re.search(r"(\d{1,3}\.\d)", odds_txt)
                if m:
                    return float(m.group(1))
    except Exception:
        pass
        
    return None

def scrape_odds(session, jcd, rno, date_str):
    # 使わないので空でOK
    return {}
