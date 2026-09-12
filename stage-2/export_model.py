import os

import torch
from gensim.models.doc2vec import Doc2Vec
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from classifier import BATCH_SIZE, EPOCHS, HIDDEN, LEARNING_RATE, load_dataset

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def main():
    embedding = Doc2Vec.load(os.path.join(DATA_DIR, "doc2vec.model"))
    x, y, boards = load_dataset(os.path.join(DATA_DIR, "tokenized.csv"), embedding)
    print(f"Vectors Ready: {len(x)} titles, {len(boards)} boards")

    torch.manual_seed(0)  # same split and init as classifier.py
    train_set, _ = random_split(TensorDataset(x, y), [0.8, 0.2])
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    net = nn.Sequential(
        nn.Linear(embedding.vector_size, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, len(boards)),
    )
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    print("Start Training")
    for epoch in range(1, EPOCHS + 1):
        net.train()
        for x_batch, y_batch in train_loader:
            loss = loss_fn(net(x_batch), y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}/{EPOCHS}")

    out_path = os.path.join(DATA_DIR, "classifier.pth")
    torch.save(
        {
            "state_dict": net.state_dict(),
            "boards": boards,
            "vector_size": embedding.vector_size,
            "hidden": HIDDEN,
        },
        out_path,
    )

    # Reload check: rebuild from the artifact alone, outputs must match
    ckpt = torch.load(out_path, map_location="cpu", weights_only=True)
    assert ckpt["boards"] == boards and ckpt["vector_size"] == embedding.vector_size
    reloaded = nn.Sequential(
        nn.Linear(ckpt["vector_size"], ckpt["hidden"]),
        nn.ReLU(),
        nn.Linear(ckpt["hidden"], len(ckpt["boards"])),
    )
    reloaded.load_state_dict(ckpt["state_dict"])
    net.eval()
    reloaded.eval()
    with torch.inference_mode():
        torch.testing.assert_close(reloaded(x[:2000]), net(x[:2000]))
    print(f"Reload check passed on 2000 vectors. Saved {out_path}")


if __name__ == "__main__":
    main()
