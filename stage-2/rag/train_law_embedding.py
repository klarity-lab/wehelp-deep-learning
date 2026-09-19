import json
import os
import sys

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

RAG_DIR = os.path.dirname(os.path.abspath(__file__))

VECTOR_SIZE = 24
MIN_COUNT = 1
EPOCHS = 400

# retrieval eval: question -> the article that actually answers it
EVAL_QUESTIONS = {
    "闖紅燈會被罰多少錢": "第53條",
    "酒駕會受到什麼處罰": "第35條",
    "騎機車沒戴安全帽罰多少": "第31條",
    "超速駕駛怎麼處罰": "第40條",
    "無照駕駛的罰則是什麼": "第21條",
}


sys.path.insert(0, RAG_DIR)
from aliases import expand
from rag import TOP_K, tokenize  # evaluate exactly what rag.py does


def hit_at_k(ranked_ids, target_article):
    # a chunk id like 第53條第2項 counts as a hit for target 第53條
    return any(cid.startswith(target_article) for cid in ranked_ids)


def main():
    chunks = json.load(open(os.path.join(RAG_DIR, "data", "chunks.json"), encoding="utf-8"))
    documents = [TaggedDocument(words=c["tokens"], tags=[i]) for i, c in enumerate(chunks)]

    print("Start Training")
    model = Doc2Vec(
        documents,
        dm=0,
        vector_size=VECTOR_SIZE,
        min_count=MIN_COUNT,
        epochs=EPOCHS,
        workers=os.cpu_count(),
    )

    # self-similarity over every chunk (corpus is tiny, no sampling needed)
    top1 = top2 = 0
    for i, c in enumerate(chunks):
        vector = model.infer_vector(c["tokens"])
        neighbors = [t for t, _ in model.dv.most_similar([vector], topn=2)]
        top1 += neighbors[0] == i
        top2 += i in neighbors
    print(f"Self Similarity {top1 / len(chunks):.3f}")
    print(f"Second Self Similarity {top2 / len(chunks):.3f}")

    # retrieval eval
    hits = 0
    print(f"\n{'question':<16} doc2vec top-{TOP_K}")
    for question, target in EVAL_QUESTIONS.items():
        q_tokens = expand(tokenize(question))

        vector = model.infer_vector(q_tokens)
        ranked_ids = [chunks[t]["id"] for t, _ in model.dv.most_similar([vector], topn=TOP_K)]
        hit = hit_at_k(ranked_ids, target)
        hits += hit

        print(f"{question:<16} {'HIT ' if hit else 'miss'} {ranked_ids}")

    print(f"\nhit@{TOP_K}: {hits}/{len(EVAL_QUESTIONS)}")

    model.save(os.path.join(RAG_DIR, "data", "law-doc2vec.model"))
    print("Model saved")


if __name__ == "__main__":
    main()
