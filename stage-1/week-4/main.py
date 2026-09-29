import math


def linear(values):
    return values


def relu(values):
    return [max(0, v) for v in values]


def sigmoid(values):
    return [1 / (1 + math.exp(-v)) for v in values]


def softmax(values):
    # Subtracting the max keeps exp() from overflowing; the result is unchanged.
    biggest = max(values)
    exps = [math.exp(v - biggest) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


# Loss functions
def mse(outputs, expects):
    return sum((e - o) ** 2 for o, e in zip(outputs, expects)) / len(outputs)


def binary_cross_entropy(outputs, expects):
    return -sum(
        e * math.log(o) + (1 - e) * math.log(1 - o) for o, e in zip(outputs, expects)
    )


def categorical_cross_entropy(outputs, expects):
    return -sum(e * math.log(o) for o, e in zip(outputs, expects))


class Network:
    def __init__(self, layers, activations):
        self.layers = layers
        self.activations = activations

    def neuron_output(self, weights, inputs):
        # The last weight is for the bias (whose input is always 1).
        input_weights = weights[:-1]
        bias_weight = weights[-1]
        weighted_sum = sum(w * x for w, x in zip(input_weights, inputs))
        return weighted_sum + bias_weight

    def forward(self, inputs):
        values = inputs
        for layer, activation in zip(self.layers, self.activations):
            values = [self.neuron_output(neuron, values) for neuron in layer]
            values = activation(values)
        return values


# Neural Network for Regression Tasks
print("Neural Network for Regression Tasks")
nn = Network(
    [
        # hidden layer: [w_X1, w_X2, w_bias]
        [
            [0.5, 0.2, 0.3],
            [0.6, -0.6, 0.25],
        ],
        # output layer: [w_h1, w_h2, w_bias]
        [
            [0.8, -0.5, 0.6],  # O1
            [0.4, 0.5, -0.25],  # O2
        ],
    ],
    [relu, linear],
)

outputs = nn.forward([1.5, 0.5])
expects = [0.8, 1]
print("Total Loss", mse(outputs, expects))

outputs = nn.forward([0, 1])
expects = [0.5, 0.5]
print("Total Loss", mse(outputs, expects))


# Neural Network for Binary Classification Tasks
print("\nNeural Network for Binary Classification Tasks")
nn = Network(
    [
        # hidden layer: [w_X1, w_X2, w_bias]
        [
            [0.5, 0.2, 0.3],
            [0.6, -0.6, 0.25],
        ],
        # output layer: [w_h1, w_h2, w_bias]
        [
            [0.8, 0.4, -0.5],  # O1
        ],
    ],
    [relu, sigmoid],
)

outputs = nn.forward([0.75, 1.25])
expects = [1]
print("Total Loss", binary_cross_entropy(outputs, expects))

outputs = nn.forward([-1, 0.5])
expects = [0]
print("Total Loss", binary_cross_entropy(outputs, expects))


# Neural Network for Multi-Label Classification Tasks
print("\nNeural Network for Multi-Label Classification Tasks")
nn = Network(
    [
        # hidden layer: [w_X1, w_X2, w_bias]
        [
            [0.5, 0.2, 0.3],
            [0.6, -0.6, 0.25],
        ],
        # output layer: [w_h1, w_h2, w_bias]
        [
            [0.8, -0.4, 0.6],  # O1
            [0.5, 0.4, 0.5],  # O2
            [0.3, 0.75, -0.5],  # O3
        ],
    ],
    [relu, sigmoid],
)

outputs = nn.forward([1.5, 0.5])
expects = [1, 0, 1]
print("Total Loss", binary_cross_entropy(outputs, expects))

outputs = nn.forward([0, 1])
expects = [1, 1, 0]
print("Total Loss", binary_cross_entropy(outputs, expects))


# Neural Network for Multi-Class Classification Tasks
print("\nNeural Network for Multi-Class Classification Tasks")
nn = Network(
    [
        # hidden layer: [w_X1, w_X2, w_bias]
        [
            [0.5, 0.2, 0.3],
            [0.6, -0.6, 0.25],
        ],
        # output layer: [w_h1, w_h2, w_bias]
        [
            [0.8, -0.4, 0.6],  # O1
            [0.5, 0.4, 0.5],  # O2
            [0.3, 0.75, -0.5],  # O3
        ],
    ],
    [relu, softmax],
)

outputs = nn.forward([1.5, 0.5])
expects = [1, 0, 0]
print("Total Loss", categorical_cross_entropy(outputs, expects))

outputs = nn.forward([0, 1])
expects = [0, 0, 1]
print("Total Loss", categorical_cross_entropy(outputs, expects))
