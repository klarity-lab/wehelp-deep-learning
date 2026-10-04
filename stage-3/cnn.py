import copy
import os
import random
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "handwriting")
TRAIN_DIR = os.path.join(DATA_DIR, "augmented_images", "augmented_images1")
TEST_DIR = os.path.join(DATA_DIR, "handwritten-english-characters-and-digits", "combined_folder", "test")

IMAGE_SIZE = 32      # 給模型的解析度
MARGIN = 1.2         # 四周留多少白邊
VAL_ORIGINALS = 4    # 抓 10% 當驗證集 (每類 44 train 11 test 所以用 44 的 10% 大概是 4)
BATCH_SIZE = 64      # 一次拿幾張算一次梯度
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 轉灰階、裁切、轉成 tensor
def load_image(path):
    gray = Image.open(path).convert("L")
    pixels = np.array(gray)                  
    ys, xs = np.where(pixels < 128) 
    assert len(xs), f"blank image: {path}"
    left, top, right, bottom = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1

    side = int(max(right - left, bottom - top) * MARGIN)
    square = Image.new("L", (side, side), 255)
    square.paste(gray.crop((left, top, right, bottom)),
                 ((side - (right - left)) // 2, (side - (bottom - top)) // 2))
    small = square.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)

    x = torch.from_numpy(np.array(small)).float() / 255
    return 1 - x 


def load_dataset(root, classes):
    # one folder per class; filename "<class>.<original>.<copy>.augmented.png"
    images, labels, originals = [], [], []
    for label, name in enumerate(tqdm(classes, desc=os.path.basename(root))):
        for filename in sorted(os.listdir(os.path.join(root, name))):
            images.append(load_image(os.path.join(root, name, filename)))
            labels.append(label)
            originals.append((label, filename.split(".")[1]))
    x = torch.stack(images).unsqueeze(1)            # (張數, 1, 32, 32)  Conv2d 要 channel 維度
    y = torch.tensor(labels)                        # (張數,)
    return x, y, originals


# 每類從 44 個編號抽 4 個
def validation_mask(originals):
    ids_by_class = defaultdict(set)
    for label, original in originals:
        ids_by_class[label].add(original)
    rng = random.Random(0)
    held_out = {(label, original)
                for label, ids in ids_by_class.items()
                for original in rng.sample(sorted(ids), VAL_ORIGINALS)}
    return torch.tensor([key in held_out for key in originals])


def evaluate(net, loader, loss_fn):
    net.eval()
    total_loss, correct, count = 0.0, 0, 0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = net(x)                         # (batch, 62)
            total_loss += loss_fn(logits, y).item() * len(x)
            correct += (logits.argmax(dim=1) == y).sum().item()
            count += len(x)
    return total_loss / count, correct, count


def main():
    classes = sorted(os.listdir(TRAIN_DIR))         # 62 folder names; index = label
    assert sorted(os.listdir(TEST_DIR)) == classes

    print(f"Loading images (training on {DEVICE})")
    x, y, originals = load_dataset(TRAIN_DIR, classes)
    x_test, y_test, _ = load_dataset(TEST_DIR, classes)

    is_val = validation_mask(originals)
    train_set = TensorDataset(x[~is_val], y[~is_val])
    val_set = TensorDataset(x[is_val], y[is_val])
    test_set = TensorDataset(x_test, y_test)
    print(f"train {len(train_set)}, val {len(val_set)}, test {len(test_set)}")

    torch.manual_seed(0)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

    net = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),  # (1, 32, 32) -> (16, 32, 32)
        nn.ReLU(),
        nn.MaxPool2d(2),                             # -> (16, 16, 16)
        nn.Conv2d(16, 32, kernel_size=3, padding=1), # -> (32, 16, 16)
        nn.ReLU(),
        nn.MaxPool2d(2),                             # -> (32, 8, 8)
        nn.Flatten(),                                # -> (2048,)
        nn.Linear(32 * 8 * 8, 128),
        nn.ReLU(),
        nn.Linear(128, len(classes)),                # -> (62,) logits
    ).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    best_correct, best_state = -1, None
    print("Start Training")
    for epoch in range(1, EPOCHS + 1):
        net.train()
        total_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            loss = loss_fn(net(x_batch), y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(x_batch)
        _, val_correct, val_count = evaluate(net, val_loader, loss_fn)
        print(f"Epoch {epoch}/{EPOCHS}  train loss {total_loss / len(train_set):.4f}"
              f"  val accuracy {val_correct / val_count:.4f}")
        if val_correct > best_correct:               # keep the epoch that generalised best
            best_correct, best_state = val_correct, copy.deepcopy(net.state_dict())

    net.load_state_dict(best_state)
    test_loss, correct, count = evaluate(net, test_loader, loss_fn)
    print(f"Average Loss in Testing Data {test_loss:.4f}")
    print(f"Correct Rate {correct}/{count} = {correct / count:.4f}")


if __name__ == "__main__":
    main()
