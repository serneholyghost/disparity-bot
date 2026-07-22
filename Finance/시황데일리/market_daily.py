import os
import requests
import FinanceDataReader as fdr
from datetime import date, timedelta

BOT_TOKEN = os.getenv("BOT_TOKEN", "")
CHAT_ID   = os.getenv("CHAT_ID", "")

INDEXES = {
    "KS11": "KOSPI",
    "KQ11": "KOSDAQ",
}

DISPARITY_TICKERS = {
    "KS11":   "KOSPI",
    "005930": "삼성전자",
    "000660": "SK하이닉스",
}

DISPARITY_THRESHOLDS = {
    "KS11":   {"red": 125, "yellow": 110},
    "005930": {"red": 140, "yellow": 125},
    "000660": {"red": 155, "yellow": 140},
}


def fetch_recent(code, days=10):
    from_date = (date.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    return fdr.DataReader(code, from_date)


def get_index_lines():
    lines = []
    for code, name in INDEXES.items():
        try:
            df = fetch_recent(code)
            close  = int(df["Close"].iloc[-1])
            change = df["Change"].iloc[-1] * 100
            sign   = "+" if change >= 0 else ""
            suffix = " & 12MF P/E 배" if code == "KS11" else ""
            lines.append(f"{name} {close:,}pt({sign}{change:.2f}%){suffix}")
        except Exception as e:
            lines.append(f"{name} 데이터 오류({e})")
    return lines


def get_disparity_emoji(ticker, disparity):
    t = DISPARITY_THRESHOLDS[ticker]
    if disparity >= t["red"]:      return "🔴"
    elif disparity >= t["yellow"]: return "🟡"
    else:                          return "🟢"


def get_ma50(ticker):
    df = fetch_recent(ticker, days=200)
    close = df["Close"].dropna()
    if len(close) < 51:
        return None, None, None, None
    today_price = int(close.iloc[-1])
    ma50        = float(close.iloc[-50:].mean())
    prev_price  = int(close.iloc[-2])
    prev_ma50   = float(close.iloc[-51:-1].mean())
    return today_price, ma50, prev_price, prev_ma50


def get_disparity(ticker):
    try:
        today_price, ma50, prev_price, prev_ma50 = get_ma50(ticker)
        if today_price is None:
            return None

        today_disp        = (today_price / ma50) * 100
        prev_disp         = (prev_price / prev_ma50) * 100
        change_pt         = today_disp - prev_disp
        price_change_pct  = (today_price - prev_price) / prev_price * 100

        return {
            "price":            today_price,
            "ma50":             round(ma50),
            "disparity":        round(today_disp, 2),
            "change_pt":        round(change_pt, 2),
            "price_change_pct": round(price_change_pct, 2),
        }
    except Exception as e:
        print(f"{ticker} 오류: {e}")
        return None


def get_disparity_lines():
    lines = []
    for ticker, name in DISPARITY_TICKERS.items():
        r = get_disparity(ticker)
        if not r:
            lines.append(f"{name}: 데이터 오류")
            continue
        emoji  = get_disparity_emoji(ticker, r["disparity"])
        sign_p = "+" if r["price_change_pct"] >= 0 else ""
        sign_d = "+" if r["change_pt"] >= 0 else ""
        lines.append(
            f"{emoji} {name}\n"
            f"현재가: {r['price']:,} ({sign_p}{r['price_change_pct']}%)\n"
            f"50일MA: {r['ma50']:,}\n"
            f"이격도: {r['disparity']}% ({sign_d}{r['change_pt']}%pt)"
        )
    return lines


def send_daily():
    today = date.today()
    lines = [f"[한국 주식 시황 {today.year}년 {today.month}월 {today.day}일 데일리]"]
    lines += get_index_lines()
    lines += ["[주요 시황]", "", "[이격도]"]
    lines += get_disparity_lines()
    lines += [
        "",
        "[특징 종목]",
        "52주 신고가: ",
        "그 외 관심 종목: ",
        "",
        "[개인 코멘트]",
    ]

    msg = "\n".join(lines)
    requests.post(
        f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
        json={"chat_id": CHAT_ID, "text": msg, "disable_web_page_preview": True},
    )
    print(msg)


if __name__ == "__main__":
    send_daily()
