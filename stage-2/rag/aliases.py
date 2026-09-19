ALIASES = {
    "酒駕": ["酒精", "濃度", "駕駛"],
    "超速": ["速度", "超過", "最高", "時速"],
    "闖紅燈": ["紅燈", "號誌"],
    "無照駕駛": ["未", "領有", "執照"],
}


def expand(tokens):
    expanded = list(tokens)
    for token in tokens:
        expanded.extend(ALIASES.get(token, []))
    return expanded
