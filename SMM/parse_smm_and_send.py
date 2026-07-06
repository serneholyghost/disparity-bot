import os
import re
import sys
import requests

# ── 설정 ──────────────────────────────────────────────
TELEGRAM_BOT_TOKEN: ${{ secrets.BOT_TOKEN }}
TELEGRAM_CHAT_ID: ${{ secrets.CHAT_ID }}
RAW_DATA = os.environ.get("SMM_DATA", "")

CODE_RE = re.compile(r"^SMM-[A-Za-z0-9\-]+$")
DATE_RE = re.compile(r"^\d{2}/\d{2}/\d{4}$")
CHANGE_RE = re.compile(r"^[+\-][\d,]+\.?\d*$")


def parse_blocks(raw: str):
    """
    붙여넣은 SMM watchlist 텍스트를 상품 단위 블록으로 파싱.
    기대 패턴 (한 상품당):
      상품명
      코드 (SMM-XX-XX-001)
      "단위\t고가\t저가\t평균\t" (탭 구분, 트레일링 탭 있을 수 있음)
      등락폭 (+6.14 / -15.38)
      기준일 (06/07/2026)
    """
    lines = [l.strip() for l in raw.splitlines() if l.strip() != ""]
    products = []
    i = 0
    name = None
    while i < len(lines):
        line = lines[i]

        if CODE_RE.match(line):
            code = line
            # 코드 앞줄이 상품명
            product_name = name if name else "(이름 미상)"

            # 다음 줄: 단위/고가/저가/평균 (탭 구분)
            i += 1
            if i >= len(lines):
                break
            data_line = lines[i]
            parts = [p for p in data_line.split("\t") if p != ""]
            unit = parts[0] if len(parts) > 0 else "?"
            high = parts[1] if len(parts) > 1 else "?"
            low = parts[2] if len(parts) > 2 else "?"
            avg = parts[3] if len(parts) > 3 else "?"

            # 다음 줄: 등락폭
            change = None
            date = None
            j = i + 1
            if j < len(lines) and CHANGE_RE.match(lines[j]):
                change = lines[j]
                j += 1
            if j < len(lines) and DATE_RE.match(lines[j]):
                date = lines[j]
                j += 1

            products.append({
                "name": product_name,
                "code": code,
                "unit": unit,
                "high": high,
                "low": low,
                "avg": avg,
                "change": change,
                "date": date,
            })
            i = j
            name = None
            continue

        # 코드가 아니면 다음 상품명 후보로 저장
        name = line
        i += 1

    return products


def format_message(products):
    if not products:
        return "⚠️ 파싱된 데이터가 없어요. 붙여넣은 형식을 확인해주세요."

    lines = ["📊 *2차전지 원자재 가격 (SMM)*", ""]
    for p in products:
        arrow = "▲" if (p["change"] and p["change"].startswith("+")) else (
            "▼" if (p["change"] and p["change"].startswith("-")) else "•"
        )
        change_str = f"{arrow} {p['change']}" if p["change"] else ""
        lines.append(
            f"*{p['name']}* (`{p['code']}`)\n"
            f"  평균: {p['avg']} {p['unit']}  (고 {p['high']} / 저 {p['low']})\n"
            f"  {change_str}   기준일: {p['date'] or '-'}"
        )
        lines.append("")
    return "\n".join(lines)


def send_telegram(text: str):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    resp = requests.post(url, data={
        "chat_id": CHAT_ID,
        "text": text,
        "parse_mode": "Markdown",
    })
    resp.raise_for_status()
    return resp.json()


if __name__ == "__main__":
    products = parse_blocks(RAW_DATA)
    message = format_message(products)
    print(message)  # 로그 확인용
    result = send_telegram(message)
    print("전송 완료:", result.get("ok"))
