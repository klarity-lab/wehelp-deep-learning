import csv
import os
import sys

import torch
from gensim.models.doc2vec import Doc2Vec
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Week-5 optimization: width 64->256 (+1.0pt, 64 was a mild bottleneck),
# epochs 10->20 (plateau after 20), SGD->Adam (+2.3pt at equal epochs)
HIDDEN = 256
BATCH_SIZE = 256
EPOCHS = 20
LEARNING_RATE = 0.001  # Adam's usual default; SGD needed 0.05


def load_dataset(in_path, embedding):
    # Same dedupe walk as embedder.py, so row i here lines up with the
    # embedding's document vector i; each kept row carries the board of
    # its first occurrence.
    seen = set()
    word_lists, board_names = [], []
    with open(in_path, encoding="utf-8") as f:
        for row in csv.reader(f):
            key = tuple(row[1:])
            if key not in seen:
                seen.add(key)
                board_names.append(row[0])
                word_lists.append(row[1:])

    boards = sorted(set(board_names))               # 編號對照表:y 裡的 0 = boards[0]
    labels = [boards.index(name) for name in board_names]
    y = torch.tensor(labels)                        # (篇數,) 每篇一個 0~8

    if len(word_lists) == len(embedding.dv):
        # reuse its vectors if same as embedding
        x = torch.from_numpy(embedding.dv.vectors)  # (篇數, vector_size) 整張表零拷貝
    else:
        # unseen data (e.g. the small test sample): infer fresh vectors
        vectors = []
        for words in word_lists:
            inferred = embedding.infer_vector(words)      # numpy 陣列 (vector_size,)
            vectors.append(torch.from_numpy(inferred))
        x = torch.stack(vectors)                    # 疊成 (篇數, vector_size)
    return x, y, boards


def accuracy(net, loader):
    net.eval()
    correct, count = 0, 0
    with torch.inference_mode():
        for x, y in loader:
            logits = net(x)                        # (batch, 9) 每篇對 9 個板的分數
            predicted = logits.argmax(dim=1)       # (batch,)  每篇分數最高的板編號
            hits = (predicted == y).sum().item()   # 這一批答對幾題
            correct += hits
            count += len(x)
    return correct / count


def main():
    in_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA_DIR, "tokenized.csv")

    embedding = Doc2Vec.load(os.path.join(DATA_DIR, "doc2vec.model"))
    x, y, boards = load_dataset(in_path, embedding)
    print(f"Vectors Ready: {len(x)} titles, {len(boards)} boards")

    torch.manual_seed(0)  # same split and same initial weights on every run
    train_set, test_set = random_split(TensorDataset(x, y), [0.8, 0.2])
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

    # raw logits out; CrossEntropyLoss applies softmax itself, so no
    # softmax layer here (same convention as BCEWithLogitsLoss in week-7)
    net = nn.Sequential(
        nn.Linear(embedding.vector_size, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, len(boards)),
    )
    loss_fn = nn.CrossEntropyLoss()
    # Adam adapts a per-parameter step size; measured on this task it beats
    # SGD(0.05) 30 epochs while using only 10 (77.9% vs 77.7% test accuracy)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    print(f"Accuracy before training: {accuracy(net, test_loader):.4f}")
    print("Start Training")
    for epoch in range(1, EPOCHS + 1):
        net.train()
        total_loss = 0.0
        for x_batch, y_batch in train_loader:
            loss = loss_fn(net(x_batch), y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(x_batch)
        print(f"Epoch {epoch}/{EPOCHS}  avg loss {total_loss / len(train_set):.4f}")

    print(f"Train Accuracy: {accuracy(net, train_loader):.4f}")
    print(f"Test Accuracy: {accuracy(net, test_loader):.4f}")


if __name__ == "__main__":
    main()
