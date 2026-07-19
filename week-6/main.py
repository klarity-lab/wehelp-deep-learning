import csv
import math
import os
import random
import sys

random.seed(42)

# Datasets are not committed. Download and place them in week-6/data/:
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def linear(values):
    return values


def relu(values):
    return [max(0, v) for v in values]


def sigmoid(values):
    return [1 / (1 + math.exp(-v)) for v in values]


def linear_derivative(values):
    return [1 for v in values]


def relu_derivative(values):
    return [1 if v > 0 else 0 for v in values]


def sigmoid_derivative(values):
    outputs = sigmoid(values)
    return [o * (1 - o) for o in outputs]


DERIVATIVES = {
    linear: linear_derivative,
    relu: relu_derivative,
    sigmoid: sigmoid_derivative,
}


class MSE:
    def get_loss(self, outputs, expects):
        n = len(outputs)
        return sum((e - o) ** 2 for o, e in zip(outputs, expects)) / n

    def get_output_gradients(self, outputs, expects):
        n = len(outputs)
        return [(2 / n) * (o - e) for o, e in zip(outputs, expects)]


class BinaryCrossEntropy:
    EPS = sys.float_info.epsilon  # keep log() away from 0 when sigmoid saturates

    def get_loss(self, outputs, expects):
        return -sum(
            e * math.log(max(o, self.EPS)) + (1 - e) * math.log(max(1 - o, self.EPS))
            for o, e in zip(outputs, expects)
        )

    def get_output_gradients(self, outputs, expects):
        return [
            -(e / max(o, self.EPS)) + (1 - e) / (max(1 - o, self.EPS))
            for o, e in zip(outputs, expects)
        ]


class Network:
    def __init__(self, layers, activations):
        self.layers = layers
        self.activations = activations
        self.activation_derivatives = [DERIVATIVES[a] for a in activations]

    def neuron_output(self, weights, inputs):
        input_weights = weights[:-1]
        bias_weight = weights[-1]
        weighted_sum = sum(w * x for w, x in zip(input_weights, inputs))
        return weighted_sum + bias_weight

    def forward(self, inputs):
        self.cache = []  # per layer: (inputs, weighted_sums, outputs)
        values = inputs
        for layer, activation in zip(self.layers, self.activations):
            weighted_sums = [self.neuron_output(neuron, values) for neuron in layer]
            outputs = activation(weighted_sums)
            self.cache.append((values, weighted_sums, outputs))
            values = outputs
        return values

    def backward(self, output_gradients):
        self.gradients: list = [None] * len(self.layers)
        downstream = output_gradients  # dL/d(this layer's outputs)
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            inputs, weighted_sums, _ = self.cache[i]
            slopes = self.activation_derivatives[i](weighted_sums)
            layer_gradients = []
            upstream = [0] * len(inputs)  # dL/d(inputs), passed to the previous layer
            for neuron, downstream_grad, slope in zip(layer, downstream, slopes):
                delta = downstream_grad * slope  # dL/dz for this neuron
                # weight gradient = delta * input; bias's input is always 1
                layer_gradients.append([delta * x for x in inputs] + [delta])
                for k in range(len(inputs)):
                    upstream[k] += delta * neuron[k]
            self.gradients[i] = layer_gradients
            downstream = upstream

    def zero_grad(self, learning_rate):
        for layer, layer_gradients in zip(self.layers, self.gradients):
            for neuron, weight_gradients in zip(layer, layer_gradients):
                for k in range(len(neuron)):
                    neuron[k] -= learning_rate * weight_gradients[k]


def dense(n_in, n_out, scale=0.5):
    return [
        [random.uniform(-scale, scale) for _ in range(n_in + 1)]  # + 1 = bias
        for _ in range(n_out)
    ]


def mean_std(values):
    mean = sum(values) / len(values)
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
    return mean, std


# ---------- Task 1: Regression - Predict Weight by Gender and Height ----------
print("------ Task 1: Weight Prediction (Regression) ------")

with open(os.path.join(DATA_DIR, "gender-height-weight.csv")) as f:
    rows = list(csv.DictReader(f))

heights = [float(r["Height"]) for r in rows]
weights = [float(r["Weight"]) for r in rows]
h_mean, h_std = mean_std(heights)
w_mean, w_std = mean_std(weights)

xs = [
    [1 if r["Gender"] == "Male" else 0, (float(r["Height"]) - h_mean) / h_std]
    for r in rows
]
es = [[(w - w_mean) / w_std] for w in weights]

nn = Network(
    [
        dense(2, 8),
        dense(8, 1),
    ],
    [relu, linear],
)
loss_fn = MSE()
learning_rate = 0.01
epochs = 100


def avg_error_in_pounds(nn):
    total = 0
    for x, w in zip(xs, weights):
        predicted = nn.forward(x)[0] * w_std + w_mean  # decode back to pounds
        total += abs(predicted - w)
    return total / len(xs)


print("Average error before training:", avg_error_in_pounds(nn))

# Training Procedure
for epoch in range(epochs):
    for x, e in zip(xs, es):
        outputs = nn.forward(x)
        output_gradients = loss_fn.get_output_gradients(outputs, e)
        nn.backward(output_gradients)
        nn.zero_grad(learning_rate)

# Evaluating Procedure
print("Average error after training:", avg_error_in_pounds(nn))  # target: < 15


# ---------- Task 2: Binary Classification - Titanic Survival ----------
print("\n------ Task 2: Titanic Survival (Binary Classification) ------")

with open(os.path.join(DATA_DIR, "titanic.csv")) as f:
    rows = list(csv.DictReader(f))

known_ages = sorted(float(r["Age"]) for r in rows if r["Age"])
age_median = known_ages[len(known_ages) // 2]  # fill missing ages with the median
ages = [float(r["Age"]) if r["Age"] else age_median for r in rows]
pclasses = [int(r["Pclass"]) for r in rows]
a_mean, a_std = mean_std(ages)
p_mean, p_std = mean_std(pclasses)  # pclass is ordinal (1>2>3), standardize the number

xs = [
    [
        1 if r["Sex"] == "female" else 0,
        (int(r["Pclass"]) - p_mean) / p_std,
        (age - a_mean) / a_std,
        int(r["SibSp"]),
        int(r["Parch"]),
    ]
    for r, age in zip(rows, ages)
]
es = [[int(r["Survived"])] for r in rows]

nn = Network(
    [
        dense(5, 8),
        dense(8, 1),
    ],
    [relu, sigmoid],
)
loss_fn = BinaryCrossEntropy()
learning_rate = 0.01
epochs = 100


def correct_rate(nn):
    correct_count = 0
    threshold = 0.5
    for x, e in zip(xs, es):
        survival_status = 1 if nn.forward(x)[0] > threshold else 0
        if survival_status == e[0]:
            correct_count += 1
    return correct_count / len(xs)


print(f"Correct rate before training: {correct_rate(nn):.2%}")

# Training Procedure
for epoch in range(epochs):
    for x, e in zip(xs, es):
        outputs = nn.forward(x)
        output_gradients = loss_fn.get_output_gradients(outputs, e)
        nn.backward(output_gradients)
        nn.zero_grad(learning_rate)

# Evaluating Procedure
print(f"Correct rate after training: {correct_rate(nn):.2%}")  # target: > 75%


# ---------- Task 3: PyTorch ----------
print("\n------ Task 3: PyTorch ------")
import torch

t = torch.tensor([[2, 3, 1], [5, -2, 1]])
print("3-1 shape:", t.shape, "dtype:", t.dtype)

r = torch.rand(3, 4, 2)
print("3-2 shape:", r.shape)
print(r)

o = torch.ones(2, 1, 5)
print("3-3 shape:", o.shape)
print(o)

a = torch.tensor([[1, 2, 4], [2, 1, 3]])
b = torch.tensor([[5], [2], [1]])
print("3-4 matmul:")
print(a @ b)

c = torch.tensor([[1, 2], [2, 3], [-1, 3]])
d = torch.tensor([[5, 4], [2, 1], [1, -5]])
print("3-5 element-wise product:")
print(c * d)
