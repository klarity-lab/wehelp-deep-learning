import csv
import os
import random
import sys

from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

VECTOR_SIZE = 100
MIN_COUNT = 3  # same idea as build_user_dict.py: rare words are mostly noise
EPOCHS = 40
EVAL_SIZE = 1000
SAVE_BAR = 0.8  # assignment: save only if Second Self-Similarity > 80%


class EpochProgress(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        self.epoch += 1
        print(f"Epoch {self.epoch}/{EPOCHS}")


def main():
    in_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA_DIR, "tokenized.csv")
    model_path = os.path.join(DATA_DIR, "doc2vec.model")

    with open(in_path, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    print("Titles Ready")

    # Identical token lists (PTT template titles — "offer 請益" appears 909
    # times) would train as separate documents and structurally sink
    # self-similarity: an inferred vector cannot rank itself above its twins.
    # Keep one copy of each. Measured effect: second self-similarity 0.72 → 0.95.
    seen = set()
    documents = []
    for row in rows:
        key = tuple(row[1:])
        if key not in seen:
            seen.add(key)
            documents.append(TaggedDocument(words=row[1:], tags=[len(documents)]))
    print("Tagged Documents Ready")

    print("Start Training")
    model = Doc2Vec(
        documents,
        dm=0,  # PV-DBOW: for ~6-word titles it beats PV-DM by a wide margin (0.66 → 0.95)
        vector_size=VECTOR_SIZE,
        min_count=MIN_COUNT,
        epochs=EPOCHS,
        workers=os.cpu_count(),
        callbacks=[EpochProgress()],
    )

    print("Test Similarity")
    random.seed(0)  # same evaluation sample on every run, so scores are comparable
    sample_indices = random.sample(range(len(documents)), min(EVAL_SIZE, len(documents)))
    top1_count = top2_count = 0
    for tested, doc_index in enumerate(sample_indices):
        if tested % 100 == 0:
            print(tested)
        vector = model.infer_vector(documents[doc_index].words)
        neighbors = [tag for tag, _ in model.dv.most_similar([vector], topn=2)]
        top1_count += neighbors[0] == doc_index
        top2_count += doc_index in neighbors
    print(len(sample_indices))

    self_sim = top1_count / len(sample_indices)
    second_self_sim = top2_count / len(sample_indices)
    print("Self Similarity", self_sim)
    print("Second Self Similarity", second_self_sim)

    if second_self_sim > SAVE_BAR:
        model.save(model_path)
        print("Model saved to", model_path)
    else:
        print("Below save bar, model not saved")


if __name__ == "__main__":
    main()
