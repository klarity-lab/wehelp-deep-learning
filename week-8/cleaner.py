import csv
import os
import re

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CLEANED = "cleaned.csv"


def main():
    rows = []
    for name in sorted(os.listdir(DATA_DIR)):
        if not name.endswith(".csv") or name == CLEANED:
            continue
        with open(os.path.join(DATA_DIR, name), encoding="utf-8") as f:
            for r in csv.DictReader(f):
                title = r["title"].strip().lower()
                if title.startswith(("re:", "fw:")):
                    continue
                title = re.sub(r"^\[[^\]]*\]\s*", "", title)  # drop the [ ] tag
                if not title:
                    continue
                rows.append([title])

    with open(os.path.join(DATA_DIR, CLEANED), "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["title"])
        writer.writerows(rows)
    print(f"Saved {len(rows)} cleaned titles to {CLEANED}")


if __name__ == "__main__":
    main()
