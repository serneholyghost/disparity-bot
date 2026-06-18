import os
import requests
import pandas as pd
from datetime import date

BOT_TOKEN = os.getenv("BOT_TOKEN", "")
CHAT_ID   = os.getenv("CHAT_ID", "")

TICKERS = {
    "KOSPI":  {"code": "KOSPI",   "name": "KOSPI",     "type": "index"},
    "005930": {"code": "005930",  "name": "삼성전자",   "type": "stock"},
    "000660": {"code": "000660",  "name": "SK하이닉스", "type": "stock"},
}

THRESHOLDS = {
    "KOSPI":  {"red": 125, "yellow": 110},
    "005930": {"red": 140, "yellow": 125},
    "000660": {"red": 155, "yellow": 140},
}

def get_emoji(key, disparity):
    t = THRESHOLDS[key]
    if disparity >= t["red"]:      return "🔴"
    elif disparity >= t["yellow"]: return "🟡"
    else:                          return "🟢"

def fetch_naver(code, is_index=False):
    if is_index:
        url = f"https://finance.naver.com/sise/sise_index_day.nhn?code={code}&page=1"
    else:
        url = f"https://finance.naver.com/item/sise_day.nhn?code={code}&page=1"
    
    headers = {"User-Agent": "Mozilla/5.0"}
    rows = []
    for page in range(1, 9):
        if is_index:
            url = f"https://finance.naver.com/sise/sise_index_day.nhn?code={code}&page={page}"
        else:
            url = f"https://finance.naver.com/item/sise_day.nhn?code={code}&page={page}"
        r = requests.get(url, headers=headers)
        tables = pd.read_html(r.text)
        df = tables[0].dropna()
        rows.append(df)
    
    data = pd.concat(rows).reset_index(drop=True)
    if is_index:
        data = data[["날짜", "체결가"]].rename(columns={"체결가": "종가"})
    else:
        data = data[["날짜", "종가"]]
    data["종가"] = pd.to_numeric(data["종가"].astype(str).str.replace(",", ""), errors="coerce")
    data = data.dropna().reset_index(drop=True)
    return data

def get_disparity(key, info):
    try:
        df = fetch_naver(info["code"], is_index=(info["type"] == "index"))
        close = df["종가"].values
        if len(close) < 51:
            return None
        today_price = int(close[0])
        today_ma50  = float(close[:50].mean())
        today_disp  = (today_price / today_ma50) * 100
        prev_price  = int(close[1])
        prev_ma50   = float(close[1:51].mean())
        prev_disp   = (prev_price / prev_ma50) * 100
        change_pt   = today_disp - prev_disp
        data_date   = df["날짜"].iloc[0]
        return {
            "price":     today_price,
            "ma50":      round(today_ma50),
            "disparity": round(today_disp, 2),
            "change_pt": round(change_pt, 2),
            "date":      data_date,
        }
    except Exception as e:
        print(f"{key} 오류: {e}")
        return None

def send_alert():
    lines = [f"📊 이격도 알림\n"]
    for key, info in TICKERS.items():
        r = get_disparity(key, info)
        if not r:
            lines.append(f"{info['name']}: 데이터 오류\n")
            continue
        emoji = get_emoji(key, r["disparity"])
        sign  = "+" if r["change_pt"] >= 0 else ""
        lines.append(
            f"{emoji} {info['name']} ({r['date']} 기준)\n"
            f"현재가: {r['price']:,}\n"
            f"50일MA: {r['ma50']:,}\n"
            f"이격도: {r['disparity']}% ({sign}{r['change_pt']}%pt)\n"
        )
    msg = "\n".join(lines)
    requests.post(
        f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
        json={"chat_id": CHAT_ID, "text": msg}
    )
    print(msg)

if __name__ == "__main__":
    send_alert()