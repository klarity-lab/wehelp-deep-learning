import csv
import os
import time
from collections import Counter, defaultdict

import torch
from torch.utils.data import DataLoader
from torchvision.io import read_image, write_png
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.utils import draw_bounding_boxes

STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(STAGE_DIR, "data", "r-cnn-data", "vehicles_images")

NAMES = ["background", "Bus", "Car", "Motorcycle", "Pickup", "Truck"]  # index = model label; 0 is reserved for background
BATCH_SIZE = 2
EPOCHS = 30
LEARNING_RATE = 0.005
DRAW_THRESHOLD = 0.5   # only for the saved pictures; the reported average uses the model default (0.05)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_split(name):
    # csv: one row per box -> group rows by image; images with no rows are skipped
    rows_by_file = defaultdict(list)
    with open(os.path.join(DATA_DIR, f"{name}_labels.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows_by_file[row["filename"]].append(row)

    images, targets = [], []
    for filename in sorted(rows_by_file):
        rows = rows_by_file[filename]
        image = read_image(os.path.join(DATA_DIR, name, filename))     # uint8 (3, 640, 640)
        images.append(image.float() / 255)                             # model wants float 0~1 （規定的）
        targets.append({
            "boxes": torch.tensor([[float(r[k]) for k in ("xmin", "ymin", "xmax", "ymax")] for r in rows]),
            "labels": torch.tensor([NAMES.index(r["class"]) for r in rows]),
        })
    return images, targets


# 搬到 GPU
def to_device(images, targets):
    return ([img.to(DEVICE) for img in images],
            [{k: v.to(DEVICE) for k, v in t.items()} for t in targets])


def main():
    train_images, train_targets = load_split("train")
    test_images, test_targets = load_split("test")
    print(f"train {len(train_images)} images / {sum(len(t['boxes']) for t in train_targets)} boxes, "
          f"test {len(test_images)} images / {sum(len(t['boxes']) for t in test_targets)} boxes")

    torch.manual_seed(0)   # before the new head is built, so its random init is identical every run
    # pre-trained on COCO; swap only the final box predictor
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(NAMES))
    model.to(DEVICE)

    loader = DataLoader(list(zip(train_images, train_targets)), batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=lambda batch: tuple(zip(*batch)))
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)

    print(f"Start Training on {DEVICE}")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        start, total_loss = time.time(), 0.0
        for images, targets in loader:
            images, targets = to_device(images, targets)
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}/{EPOCHS}  loss {total_loss / len(loader):.4f}  ({time.time() - start:.0f}s)")

    # evaluate: every detection the model returns (default score threshold 0.05) counts
    model.eval()
    detections = []
    with torch.inference_mode():
        for image in test_images:
            detections.append({k: v.cpu() for k, v in model([image.to(DEVICE)])[0].items()})
    scores = torch.cat([d["scores"] for d in detections])
    labels = torch.cat([d["labels"] for d in detections])
    print(f"Detected {len(scores)} objects in {len(test_images)} test images "
          f"(score threshold {model.roi_heads.score_thresh})")
    print("Per class:", dict(Counter(NAMES[l] for l in labels.tolist())))
    print(f"Average Score {scores.mean():.4f}")

    # save two test images with boxes; keep only confident boxes so the picture stays readable
    for i in range(2):
        d = detections[i]
        keep = d["scores"] >= DRAW_THRESHOLD
        drawn = draw_bounding_boxes(
            (test_images[i] * 255).byte(), d["boxes"][keep], colors="red", width=2,
            labels=[f"{NAMES[l]} {s:.2f}" for l, s in zip(d["labels"][keep].tolist(), d["scores"][keep].tolist())],
        )
        write_png(drawn, os.path.join(STAGE_DIR, f"detected-{i + 1}.png"))
        print(f"detected-{i + 1}.png: {int(keep.sum())} boxes with score >= {DRAW_THRESHOLD}")


if __name__ == "__main__":
    main()
