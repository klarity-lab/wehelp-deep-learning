import json
import os
import re
import sys

STAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, STAGE_DIR)

import jieba
import jieba.posseg as pseg

from tokenizer import keep_token

SPLIT_OVER = 400
# 款/目 markers: lines that CONTINUE the current 項 rather than starting a new one
CONTINUATION = re.compile(r"^(（[一二三四五六七八九十]+）|[一二三四五六七八九十]+、|\d+\.)")


def split_paragraphs(content):
    """Split an article body into 項 (paragraphs), keeping 款/目 lines attached."""
    paragraphs = []
    for line in content.replace("\r\n", "\n").split("\n"):
        if not line.strip():
            continue
        if CONTINUATION.match(line.strip()) and paragraphs:
            paragraphs[-1] += "\n" + line
        else:
            paragraphs.append(line)
    return paragraphs


def main():
    jieba.setLogLevel(60)
    jieba.set_dictionary(os.path.join(STAGE_DIR, "data", "dict-zh-tw.txt"))
    tagger = pseg.POSTokenizer(jieba.dt)

    law = json.load(open(os.path.join(STAGE_DIR, "data", "traffic-law.json"), encoding="utf-8"))

    chunks = []
    for article in law["LawArticles"]:
        number = article["ArticleNo"].replace(" ", "")  # "第 92-2 條" -> "第92-2條"
        content = article["ArticleContent"].strip()
        if not number or "（刪除）" in content:
            continue

        if len(content) > SPLIT_OVER:
            parts = [
                (f"{number}第{i}項", paragraph)
                for i, paragraph in enumerate(split_paragraphs(content), 1)
            ]
        else:
            parts = [(number, content)]

        for chunk_id, text in parts:
            tokens = [w for w, tag in tagger.cut(text) if keep_token(w, tag)]
            chunks.append({"id": chunk_id, "text": f"{number}:{text}", "tokens": tokens})

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "chunks.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=1)

    sizes = [len(c["text"]) for c in chunks]
    print(f"{len(chunks)} chunks (from {len(law['LawArticles'])} articles), "
          f"chars min/median/max = {min(sizes)}/{sorted(sizes)[len(sizes)//2]}/{max(sizes)}")


if __name__ == "__main__":
    main()
