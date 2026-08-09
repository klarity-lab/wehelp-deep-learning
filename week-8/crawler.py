import asyncio
import csv
import html
import os
import re

import httpx

BOARDS = [
    "baseball",
    "Boy-Girl",
    "c_chat",
    "hatepolitics",
    "Lifeismoney",
    "Military",
    "pc_shopping",
    "stock",
    "Tech_Job",
]
MAX_TITLES = 100_000
CONCURRENCY = 8  
BATCH = 200
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


async def fetch(client, url):
    for attempt in range(3):
        try:
            response = await client.get(url, timeout=10)
            response.raise_for_status()
            return response.text
        except httpx.HTTPError:
            if attempt == 2:
                raise
            await asyncio.sleep(2)


def parse_titles(page):
    # pinned posts sit below this separator on the newest page; deleted
    # articles have no <a> inside their title div, so the regex skips them
    page = page.split('<div class="r-list-sep"', 1)[0]
    return [
        html.unescape(t)
        for t in re.findall(r'<div class="title">\s*<a href="[^"]*">([^<]*)</a>', page)
    ]


async def crawl_board(client, board):
    page = await fetch(client, f"https://www.ptt.cc/bbs/{board}/index.html")
    # the「上頁」button links to index<N>.html, so the newest page is N + 1
    newest = int(re.search(r'href="/bbs/[^"]+/index(\d+)\.html">&lsaquo;', page).group(1)) + 1

    titles = []
    numbers = list(range(newest, 0, -1))
    for i in range(0, len(numbers), BATCH):
        pages = await asyncio.gather(
            *(
                fetch(client, f"https://www.ptt.cc/bbs/{board}/index{n}.html")
                for n in numbers[i : i + BATCH]
            )
        )
        for p in pages:
            titles.extend(parse_titles(p))
        print(f"  {board}: {min(i + BATCH, len(numbers))}/{len(numbers)} pages, {len(titles)} titles")
        if len(titles) >= MAX_TITLES:
            break
    return titles[:MAX_TITLES]


async def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    async with httpx.AsyncClient(
        headers={"User-Agent": "Mozilla/5.0"},
        cookies={"over18": "1"},
        limits=httpx.Limits(max_connections=CONCURRENCY),
    ) as client:
        for board in BOARDS:
            print(f"Crawling {board} ...")
            titles = await crawl_board(client, board)
            with open(
                os.path.join(DATA_DIR, f"{board}.csv"), "w", encoding="utf-8", newline=""
            ) as f:
                writer = csv.writer(f, lineterminator="\n")
                writer.writerow(["board", "title"])
                writer.writerows([board, title] for title in titles)
            print(f"{board}: saved {len(titles)} titles")


if __name__ == "__main__":
    asyncio.run(main())
