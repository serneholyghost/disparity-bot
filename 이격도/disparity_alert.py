import os
import requests
from pykrx import stock
from datetime import date, timedelta

BOT_TOKEN = os.getenv("BOT_TOKEN", "")
CHAT_ID   = os.getenv("CHAT_ID", "")

TICKERS = {
    "1001":   "KOSPI",
    "005930": "삼성전자",
    "000660": "SK하이닉스",
}

THRESHOLDS = {
    "1001":   {"red": 125, "yellow": 110},
    "005930": {"red": 140, "yellow": 125},
    "000660": {"red": 155, "yellow": 140},
}

def get_emoji(ticker, disparity):
    t = THRESHOLDS[ticker]
    if disparity >= t["red"]:      return "🔴"
    elif disparity >= t["yellow"]: return "🟡"
    else:                          return "🟢"

def get_disparity(ticker):
    from_date = (date.today() - timedelta(days=200)).strftime("%Y%m%d")
    to_date   = date.today().strftime("%Y%m%d")

    if ticker == "1001":
        df = stock.get_index_ohlcv_by_date(from_date, to_date, "1001")
        close = df["종가"].dropna()
    else:
        df = stock.get_market_ohlcv_by_date(from_date, to_date, ticker)
        close = df["종가"].dropna()

    if len(close) < 51:
        return None

    today_price = int(close.iloc[-1])
    today_ma50  = float(close.iloc[-50:].mean())
    today_disp  = (today_price / today_ma50) * 100

    prev_price  = int(close.iloc[-2])
    prev_ma50   = float(close.iloc[-51:-1].mean())
    prev_disp   = (prev_price / prev_ma50) * 100

    change_pt   = today_disp - prev_disp

    return {
        "price":     today_price,
        "ma50":      round(today_ma50),
        "disparity": round(today_disp, 2),
        "change_pt": round(change_pt, 2),
        "date":      close.index[-1].strftime("%m/%d"),
    }

def send_alert():
    lines = [f"📊 이격도 알림 ({date.today().strftime('%m/%d')} 종가 기준)\n"]
    for ticker, name in TICKERS.items():
        r = get_disparity(ticker)
        if not r:
            lines.append(f"{name}: 데이터 오류\n")
            continue
        emoji = get_emoji(ticker, r["disparity"])
        sign  = "+" if r["change_pt"] >= 0 else ""
        lines.append(
            f"{emoji} {name}\n"
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