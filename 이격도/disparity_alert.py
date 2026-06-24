import os
import requests
import yfinance as yf
import FinanceDataReader as fdr
from datetime import date, timedelta

BOT_TOKEN = os.getenv("BOT_TOKEN", "")
CHAT_ID   = os.getenv("CHAT_ID", "")

TICKERS = {
    "KS11":   {"name": "KOSPI",     "yf": "^KS11"},
    "005930": {"name": "삼성전자",   "yf": "005930.KS"},
    "000660": {"name": "SK하이닉스", "yf": "000660.KS"},
}

THRESHOLDS = {
    "KS11":   {"red": 125, "yellow": 110},
    "005930": {"red": 140, "yellow": 125},
    "000660": {"red": 155, "yellow": 140},
}

def get_emoji(ticker, disparity):
    t = THRESHOLDS[ticker]
    if disparity >= t["red"]:        return "🔴"
    elif disparity >= t["yellow"]:   return "🟡"
    else:                            return "🟢"

def get_current_price(yf_ticker):
    tk = yf.Ticker(yf_ticker)
    data = tk.history(period="1d", interval="1m")
    if data.empty:
        return None
    return float(data["Close"].iloc[-1])

def get_ma50(fdr_ticker):
    from_date = (date.today() - timedelta(days=200)).strftime("%Y-%m-%d")
    df = fdr.DataReader(fdr_ticker, from_date)
    close = df["Close"].dropna()
    if len(close) < 51:
        return None, None
    ma50      = float(close.iloc[-50:].mean())
    prev_ma50 = float(close.iloc[-51:-1].mean())
    prev_price = int(close.iloc[-2])
    return ma50, prev_ma50, prev_price

def get_disparity(ticker, info):
    try:
        current_price = get_current_price(info["yf"])
        if not current_price:
            return None

        result = get_ma50(ticker)
        if result[0] is None:
            return None
        ma50, prev_ma50, prev_price = result

        today_disp       = (current_price / ma50) * 100
        prev_disp        = (prev_price / prev_ma50) * 100
        change_pt        = today_disp - prev_disp
        price_change_pct = (current_price - prev_price) / prev_price * 100

        return {
            "price":            int(current_price),
            "ma50":             round(ma50),
            "disparity":        round(today_disp, 2),
            "change_pt":        round(change_pt, 2),
            "price_change_pct": round(price_change_pct, 2),
        }
    except Exception as e:
        print(f"{ticker} 오류: {e}")
        return None


def send_alert():
    now = date.today().strftime("%m/%d")
    lines = [f"📊 이격도 알림 ({now} 기준)\n"]
    for ticker, info in TICKERS.items():
        r = get_disparity(ticker, info)
        if not r:
            lines.append(f"{info['name']}: 데이터 오류\n")
            continue
        emoji  = get_emoji(ticker, r["disparity"])
        sign_p = "+" if r["price_change_pct"] >= 0 else ""
        sign_d = "+" if r["change_pt"] >= 0 else ""
        lines.append(
            f"{emoji} {info['name']}  {sign_p}{r['price_change_pct']}%\n"
            f"현재가: {r['price']:,}  50일MA: {r['ma50']:,}\n"
            f"이격도: {r['disparity']}% ({sign_d}{r['change_pt']}%pt)\n"
        )
    msg = "\n".join(lines)
    requests.post(
        f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
        json={"chat_id": CHAT_ID, "text": msg}
    )
    print(msg)

if __name__ == "__main__":
    send_alert()
