import asyncio
import csv
import hashlib
import os
import re
import sys

import jieba
import jieba.posseg as pseg
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from gensim.models.doc2vec import Doc2Vec
from pydantic import BaseModel
from torch import nn

APP_DIR = os.path.dirname(os.path.abspath(__file__))
STAGE_DIR = os.path.dirname(APP_DIR)
DATA_DIR = os.path.join(STAGE_DIR, "data")
FEEDBACK_CSV = os.path.join(STAGE_DIR, "user-labeled-titles.csv")
MAX_TITLE_LEN = 200

sys.path.insert(0, STAGE_DIR)
from tokenizer import keep_token  # single source of truth for POS filtering

# ---- preload everything at startup: first request must be fast ----
jieba.setLogLevel(60)
jieba.set_dictionary(os.path.join(DATA_DIR, "dict-zh-tw.txt"))
jieba.load_userdict(os.path.join(DATA_DIR, "user-dict.txt"))
tagger = pseg.POSTokenizer(jieba.dt)
tagger.cut("預熱")  # force prefix-dict build now, not on first request

embedding = Doc2Vec.load(os.path.join(DATA_DIR, "doc2vec.model"), mmap="r")
ckpt = torch.load(os.path.join(DATA_DIR, "classifier.pth"), map_location="cpu", weights_only=True)
BOARDS = ckpt["boards"]
net = nn.Sequential(
    nn.Linear(ckpt["vector_size"], ckpt["hidden"]),
    nn.ReLU(),
    nn.Linear(ckpt["hidden"], len(BOARDS)),
)
net.load_state_dict(ckpt["state_dict"])
net.eval()

infer_lock = asyncio.Lock()  # guards RNG seeding + infer_vector as one unit
feedback_lock = asyncio.Lock()

app = FastAPI()


def clean_and_tokenize(title):
    # mirror cleaner.py rules; serving-side choice: a pasted "re: ..." title
    # gets its prefix stripped and classified (training dropped such rows)
    title = title.strip().lower()
    title = re.sub(r"^(re:|fw:)\s*", "", title)
    title = re.sub(r"^\[[^\]]*\]\s*", "", title)
    return [word for word, tag in tagger.cut(title) if keep_token(word, tag)]


def classify(vector):
    with torch.inference_mode():
        logits = net(torch.from_numpy(vector))
    return BOARDS[logits.argmax().item()]


@app.get("/")
def home():
    return FileResponse(os.path.join(APP_DIR, "static", "index.html"))


@app.get("/api/model/prediction")
async def prediction(title: str):
    if not title.strip() or len(title) > MAX_TITLE_LEN:
        raise HTTPException(status_code=400, detail="title must be 1-200 characters")
    words = clean_and_tokenize(title)
    if not words:
        raise HTTPException(status_code=422, detail="nothing left to classify after filtering")

    # same title -> same seed -> same prediction on every call
    seed = int(hashlib.md5(" ".join(words).encode()).hexdigest()[:8], 16)
    async with infer_lock:
        embedding.random.seed(seed)
        vector = embedding.infer_vector(words)
    return {"label": classify(vector)}


class Feedback(BaseModel):
    title: str
    label: str


@app.post("/api/model/feedback")
async def feedback(item: Feedback):
    if not item.title.strip() or len(item.title) > MAX_TITLE_LEN:
        raise HTTPException(status_code=400, detail="title must be 1-200 characters")
    if item.label not in BOARDS:
        raise HTTPException(status_code=400, detail="unknown label")

    async with feedback_lock:
        is_new = not os.path.exists(FEEDBACK_CSV)
        with open(FEEDBACK_CSV, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            if is_new:
                writer.writerow(["title", "label"])
            writer.writerow([item.title.strip(), item.label])
    return {"message": "ok"}
