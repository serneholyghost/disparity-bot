import os
import requests
import FinanceDataReader as fdr
import xml.etree.ElementTree as ET
from datetime import date, timedelta

BOT_TOKEN = os.getenv("BOT_TOKEN", "")
CHAT_ID   = os.getenv("CHAT_ID", "")

INDEXES = {
    "KS11": "코스피",
    "KQ11": "코스닥",
}

NEWS_RSS_URL = "https://www.hankyung.com/feed/finance"
NEWS_COUNT = 5


def fetch_recent(code, days=10):
    from_date = (date.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    return fdr.DataReader(code, from_date)


def get_index_line():
    parts = []
    for code, name in INDEXES.items():
        try:
            df = fetch_recent(code)
            close  = int(df["Close"].iloc[-1])
            change = df["Change"].iloc[-1] * 100
            sign   = "+" if change >= 0 else ""
            parts.append(f"{name} {close:,}({sign}{change:.2f}%)")
        except Exception as e:
            parts.append(f"{name} 데이터 오류({e})")
    return "  ".join(parts)


def get_news_lines():
    lines = ["주요 뉴스"]
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(NEWS_RSS_URL, headers=headers, timeout=10)
        r.raise_for_status()
        r.encoding = "utf-8"
        root = ET.fromstring(r.content)
        items = root.findall(".//item")[:NEWS_COUNT]
        for i, item in enumerate(items, 1):
            title = (item.findtext("title") or "").strip()
            link  = (item.findtext("link") or "").strip()
            lines.append(f"{i}. {title}\n{link}")
    except Exception as e:
        lines.append(f"뉴스를 가져오지 못했습니다: {e}")
    return lines


def send_daily():
    today = date.today().strftime("%Y/%m/%d")
    lines = [
        f"📊 한국 주식 시황 데일리 ({today})",
        "",
        get_index_line(),
        "",
    ]
    lines += get_news_lines()

    msg = "\n".join(lines)
    requests.post(
        f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
        json={"chat_id": CHAT_ID, "text": msg, "disable_web_page_preview": True},
    )
    print(msg)


if __name__ == "__main__":
    send_daily()
