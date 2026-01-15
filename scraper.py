import requests
from bs4 import BeautifulSoup
import re

def fetch_odds_nirentan(jcd, rno, date):
    """二連単の全オッズを取得して辞書で返す"""
    url = f"https://www.boatrace.jp/owpc/pc/race/odds2t?rno={rno}&jcd={jcd}&hd={date}"
    odds_dict = {}
    try:
        res = requests.get(url, timeout=10)
        res.encoding = res.apparent_encoding
        soup = BeautifulSoup(res.text, 'html.parser')
        
        # 二連単のテーブル（1号艇が頭、2号艇が頭...と分かれている）
        tables = soup.select('table.is-p_auto')
        for first_boat, table in enumerate(tables, 1):
            rows = table.select('tbody tr')
            for row in rows:
                cols = row.select('td')
                if len(cols) >= 2:
                    second_boat = cols[0].text.strip()
                    odds_val = cols[1].text.strip()
                    if odds_val and odds_val != '-':
                        odds_dict[f"{first_boat}-{second_boat}"] = float(odds_val)
        return odds_dict
    except Exception as e:
        print(f"Odds Error: {e}")
        return None

def fetch_race_result(jcd, rno, date):
    """結果（買い目）と二連単の配当（100円あたり）を取得"""
    url = f"https://www.boatrace.jp/owpc/pc/race/raceresult?rno={rno}&jcd={jcd}&hd={date}"
    try:
        res = requests.get(url, timeout=10)
        res.encoding = res.apparent_encoding
        soup = BeautifulSoup(res.text, 'html.parser')
        
        # 払戻金テーブル（2番目のis-w600テーブルが配当表であることが多い）
        payout_tables = soup.select('table.is-w600')
        if not payout_tables:
            return None, None # まだ結果が出ていない
            
        winning_combo = None
        payout = None
        
        # 配当表の中から「2連単」の行を探す
        for table in payout_tables:
            rows = table.select('tr')
            for row in rows:
                if '2連単' in row.text:
                    tds = row.select('td')
                    # 例: [買い目, 払戻金, 人気] の順
                    winning_combo = tds[0].text.strip().replace(' ', '')
                    payout_text = tds[1].text.strip().replace('¥', '').replace(',', '')
                    payout = int(payout_text)
                    return winning_combo, payout
        return None, None
    except Exception as e:
        print(f"Result Error: {e}")
        return None, None