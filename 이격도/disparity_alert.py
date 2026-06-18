import os
import requests
import yfinance as yf
from datetime import date, timedelta

BOT_TOKEN = os.getenv("BOT_TOKEN", "")
CHAT_ID   = os.getenv("CHAT_ID", "")

TICKERS = {
    "^KS11":     "KOSPI",
    "005930.KS": "삼성전자",
    "000660.KS": "SK하이닉스",
}

THRESHOLDS = {
    "^KS11":     {"red": 125, "yellow": 110},
    "005930.KS": {"red": 140, "yellow": 125},
    "000660.KS": {"red": 155, "yellow": 140},
}

def get_emoji(ticker, disparity):
    t = THRESHOLDS[ticker]
    if disparity >= t["red"]:    return "🔴"
    elif disparity >= t["yellow"]: return "🟡"
    else:                          return "🟢"

def get_disparity(ticker):
    from_date = (date.today() - timedelta(days=200)).strftime("%Y-%m-%d")
    df = yf.download(ticker, start=from_date, auto_adjust=True, progress=False)
    if hasattr(df.columns, "levels"):
        df.columns = df.columns.get_level_values(0)
    close = df["Close"].dropna()
    if len(close) < 51:
        return None

    # 오늘 이격도
    today_price    = float(close.iloc[-1])
    today_ma50     = float(close.iloc[-50:].mean())
    today_disp     = (today_price / today_ma50) * 100

    # 전일 이격도
    prev_price     = float(close.iloc[-2])
    prev_ma50      = float(close.iloc[-51:-1].mean())
    prev_disp      = (prev_price / prev_ma50) * 100

    change_pt      = today_disp - prev_disp

    return {
        "price":     int(today_price),
        "ma50":      round(today_ma50),
        "disparity": round(today_disp, 2),
        "change_pt": round(change_pt, 2),
    }

def send_alert():
    lines = [f"📊 이격도 알림 ({date.today().strftime('%m/%d')} 종가 기준)\n"]
    for ticker, name in TICKERS.items():
        r = get_disparity(ticker)
        if not r:
            lines.append(f"{name}: 데이터 오류\n")
            continue
        emoji  = get_emoji(ticker, r["disparity"])
        sign   = "+" if r["change_pt"] >= 0 else ""
        lines.append(
            f"{emoji} {name}\n"
            f"  현재가: {r['price']:,}\n"
            f"  50일MA: {r['ma50']:,}\n"
            f"  이격도: {r['disparity']}%  ({sign}{r['change_pt']}%pt)\n"
        )
    msg = "\n".join(lines)
    requests.post(
        f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
        json={"chat_id": CHAT_ID, "text": msg}
    )
    print("전송 완료\n", msg)

if __name__ == "__main__":
    send_alert()