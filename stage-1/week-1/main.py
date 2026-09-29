import csv
import json
import urllib.request
from pathlib import Path

# pageCount=40 is optional but mirrors what PChome's own frontend sends,
# keeping us on the request path the site itself exercises and maintains
API_URL = "https://ecshweb.pchome.com.tw/search/v4.3/all/results?cateid=DSAA31&pageCount=40&page={page}"


def fetch_page(page):
    request = urllib.request.Request(
        API_URL.format(page=page),
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_all_products():
    products = []
    page = 1
    while True:
        data = fetch_page(page)
        products.extend(data.get("Prods") or [])
        if page >= data.get("TotalPage", 0):
            break
        page += 1
    return products


def main():
    products = fetch_all_products()

    # Task 1: all product IDs
    Path("products.txt").write_text(
        "".join(p["Id"] + "\n" for p in products), encoding="utf-8"
    )

    # Task 2: at least 1 review and average rating > 4.9
    # "or 0" needed: the API sends "ratingValue": null (not a missing key)
    # for unreviewed products, so .get() alone would return None
    best_ids = [
        p["Id"]
        for p in products
        if (p.get("reviewCount") or 0) >= 1 and (p.get("ratingValue") or 0) > 4.9
    ]
    Path("best-products.txt").write_text(
        "".join(product_id + "\n" for product_id in best_ids), encoding="utf-8"
    )

    # Task 3: average price of PCs with Intel i5 processor
    i5_prices = [p["Price"] for p in products if "i5" in p["Name"].lower()]
    average_i5_price = sum(i5_prices) / len(i5_prices)
    print(average_i5_price)

    # Task 4: z-score standardization of prices (population)
    prices = [p["Price"] for p in products]
    mean = sum(prices) / len(prices)
    std = (sum((price - mean) ** 2 for price in prices) / len(prices)) ** 0.5
    with Path("standardization.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        for product in products:
            z_score = (product["Price"] - mean) / std
            writer.writerow([product["Id"], product["Price"], round(z_score, 4)])


if __name__ == "__main__":
    main()
