"""Offline vocabulary mining: run CKIP NER over the cleaned titles and collect
frequent named entities into a jieba user dictionary, so the fast jieba pass in
tokenizer.py can segment PTT-specific names (people, teams, anime titles...)
that its dictionary does not know."""

import csv
import os
import re
from collections import Counter, defaultdict

from ckip_transformers.nlp import CkipNerChunker

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# NER type -> jieba POS tag, so posseg keeps them out of the drop list
KEEP_TYPES = {"PERSON": "nr", "ORG": "nt", "WORK_OF_ART": "nz", "PRODUCT": "nz"}
MIN_COUNT = 3  # an entity seen fewer times is likely a mis-recognition


def main():
    with open(os.path.join(DATA_DIR, "cleaned.csv"), encoding="utf-8") as f:
        titles = [r["title"] for r in csv.DictReader(f)]

    ner = CkipNerChunker(model="bert-base")
    counter = Counter()
    tag_votes = defaultdict(Counter)  # NER may type the same word differently
    for spans in ner(titles, show_progress=True):
        for span in spans:
            word = span.word.strip()
            # only pure-Chinese words of 2-6 chars: jieba already handles
            # ASCII well, and longer spans are usually NER noise
            if span.ner in KEEP_TYPES and re.fullmatch(r"[一-鿿]{2,6}", word):
                counter[word] += 1
                tag_votes[word][KEEP_TYPES[span.ner]] += 1

    words = sorted((w, tag_votes[w].most_common(1)[0][0]) for w, c in counter.items() if c >= MIN_COUNT)
    with open(os.path.join(DATA_DIR, "user-dict.txt"), "w", encoding="utf-8") as f:
        for word, tag in words:
            f.write(f"{word} {tag}\n")  # no freq: jieba picks one high enough
    print(f"Saved {len(words)} words to user-dict.txt")


if __name__ == "__main__":
    main()
