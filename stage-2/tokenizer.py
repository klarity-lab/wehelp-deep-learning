import csv
import os
import sys

import jieba
import jieba.posseg as pseg
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Drop function words — they carry no topic information. Two tag sets coexist:
# words from dict-zh-tw.txt carry CKIP-style uppercase tags, words guessed by
# jieba's HMM carry lowercase jieba tags (with subtypes: 的=uj, 了=ul...).
DROP_TAGS = {"P", "C", "T", "ASP", "POST"}  # 介係詞 連接詞 語助詞 動貌詞 後置詞
DROP_FAMILIES = "pcuxey"  # 介係詞 連接詞 助詞 標點/未知 嘆詞 語氣詞


def keep_token(word, tag):
    if not word.strip() or tag in DROP_TAGS:
        return False
    # "eng" (English) must not be caught by the lowercase e-family check
    return tag == "eng" or tag[0] not in DROP_FAMILIES


def main():
    in_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA_DIR, "cleaned.csv")
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(DATA_DIR, "tokenized.csv")

    # Traditional Chinese main dictionary (from jieba-zh_TW), plus the
    # PTT-specific names mined by build_user_dict.py when available
    jieba.set_dictionary(os.path.join(DATA_DIR, "dict-zh-tw.txt"))
    user_dict = os.path.join(DATA_DIR, "user-dict.txt")
    if os.path.exists(user_dict):
        jieba.load_userdict(user_dict)

    # default pseg reads POS tags from jieba's bundled dictionary, so words
    # that only exist in dict-zh-tw.txt would all be mistagged as x and
    # dropped as punctuation; bind the tagger to our tokenizer instead
    tagger = pseg.POSTokenizer(jieba.dt)

    rows = []
    with open(in_path, encoding="utf-8") as f:
        for row in tqdm(list(csv.DictReader(f)), desc="Tokenizing"):
            words = [word for word, tag in tagger.cut(row["title"]) if keep_token(word, tag)]
            if words:
                rows.append([row["board"]] + words)

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows(rows)
    print(f"Saved {len(rows)} tokenized titles to {os.path.basename(out_path)}")


if __name__ == "__main__":
    main()
