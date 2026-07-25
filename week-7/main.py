import csv
import os

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def mean_std(values):
    mean = sum(values) / len(values)
    std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
    return mean, std


class MyData(Dataset):
    def __init__(self, xs, es):
        self.xs = torch.tensor(xs, dtype=torch.float32)
        self.es = torch.tensor(es, dtype=torch.float32)

    def __getitem__(self, index):
        return self.xs[index], self.es[index]

    def __len__(self):
        return len(self.xs)


def split_train_eval(rows):
    # Split the RAW rows in 4:1 ratio
    train_rows, eval_rows = random_split(rows, [0.8, 0.2])
    return list(train_rows), list(eval_rows)


def train(model, loss_fn, loader, learning_rate, epochs):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    for _ in range(epochs):
        model.train()
        for x, e in loader:
            loss = loss_fn(model(x), e)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


# ---------- Task 1: Regression - Predict Weight by Gender and Height ----------
print("------ Task 1: Weight Prediction (Regression) ------")

with open(os.path.join(DATA_DIR, "gender-height-weight.csv")) as f:
    rows = list(csv.DictReader(f))

train_rows, eval_rows = split_train_eval(rows)

# Fit standardization on TRAIN rows
h_mean, h_std = mean_std([float(r["Height"]) for r in train_rows])
w_mean, w_std = mean_std([float(r["Weight"]) for r in train_rows])


def make_regression_data(rs):
    xs = [[1 if r["Gender"] == "Male" else 0, (float(r["Height"]) - h_mean) / h_std] for r in rs]
    es = [[(float(r["Weight"]) - w_mean) / w_std] for r in rs]
    return xs, es


train_loader = DataLoader(MyData(*make_regression_data(train_rows)), batch_size=64, shuffle=True)
eval_loader = DataLoader(MyData(*make_regression_data(eval_rows)), batch_size=64)

model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))
loss_fn = nn.MSELoss()


def avg_error_in_pounds(model, loader):
    model.eval()
    total, count = 0.0, 0
    with torch.inference_mode():
        for x, e in loader:
            predicted_pounds = model(x) * w_std + w_mean  # decode back to pounds
            actual_pounds = e * w_std + w_mean
            total += (predicted_pounds - actual_pounds).abs().sum().item()
            count += len(x)
    return total / count


print("Average error before training:", avg_error_in_pounds(model, eval_loader))
train(model, loss_fn, train_loader, learning_rate=0.01, epochs=100)
print("Average error after training:", avg_error_in_pounds(model, eval_loader))  # target: < 15


# ---------- Task 2: Binary Classification - Titanic Survival ----------
print("\n------ Task 2: Titanic Survival (Binary Classification) ------")

with open(os.path.join(DATA_DIR, "titanic.csv")) as f:
    rows = list(csv.DictReader(f))

train_rows, eval_rows = split_train_eval(rows)

# Fit everything (missing-age fill value + standardization stats) on TRAIN rows only.
known_ages = sorted(float(r["Age"]) for r in train_rows if r["Age"])
age_median = known_ages[len(known_ages) // 2]


def age_of(r):
    return float(r["Age"]) if r["Age"] else age_median


a_mean, a_std = mean_std([age_of(r) for r in train_rows])
p_mean, p_std = mean_std([int(r["Pclass"]) for r in train_rows])  # pclass ordinal (1>2>3)


def make_titanic_data(rs):
    xs = [
        [
            1 if r["Sex"] == "female" else 0,
            (int(r["Pclass"]) - p_mean) / p_std,
            (age_of(r) - a_mean) / a_std,
            int(r["SibSp"]),
            int(r["Parch"]),
        ]
        for r in rs
    ]
    es = [[int(r["Survived"])] for r in rs]
    return xs, es


train_loader = DataLoader(MyData(*make_titanic_data(train_rows)), batch_size=16, shuffle=True)
eval_loader = DataLoader(MyData(*make_titanic_data(eval_rows)), batch_size=16)

# Output raw logits; BCEWithLogitsLoss folds in the sigmoid (numerically stable).
model = nn.Sequential(nn.Linear(5, 8), nn.ReLU(), nn.Linear(8, 1))
loss_fn = nn.BCEWithLogitsLoss()


def correct_rate(model, loader):
    model.eval()
    correct, count = 0, 0
    with torch.inference_mode():
        for x, e in loader:
            predicted = (model(x) > 0).float()  # logit > 0  <=>  probability > 0.5
            correct += (predicted == e).sum().item()
            count += len(x)
    return correct / count


print(f"Correct rate before training: {correct_rate(model, eval_loader):.2%}")
train(model, loss_fn, train_loader, learning_rate=0.01, epochs=100)
print(f"Correct rate after training: {correct_rate(model, eval_loader):.2%}")  # target: > 75%
