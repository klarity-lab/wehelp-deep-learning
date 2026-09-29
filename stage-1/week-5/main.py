import math


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
    def get_loss(self, outputs, expects):
        return -sum(
            e * math.log(o) + (1 - e) * math.log(1 - o)
            for o, e in zip(outputs, expects)
        )

    def get_output_gradients(self, outputs, expects):
        return [-(e / o) + (1 - e) / (1 - o) for o, e in zip(outputs, expects)]


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


def build_regression_network():
    return Network(
        [
            [[0.5, 0.2, 0.3], [0.6, -0.6, 0.25]],  # hidden layer 1, ReLU
            [[0.8, -0.5, 0.6]],  # hidden layer 2, Linear
            [[0.6, 0.4], [-0.3, 0.75]],  # output layer, Linear
        ],
        [relu, linear, linear],
    )


def build_binary_network():
    return Network(
        [
            [[0.5, 0.2, 0.3], [0.6, -0.6, 0.25]],  # hidden layer, ReLU
            [[0.8, 0.4, -0.5]],  # output layer, Sigmoid
        ],
        [relu, sigmoid],
    )


def print_new_weights(nn):
    # weights and biases on separate lines, one row per layer
    for i, layer in enumerate(nn.layers):
        print(f"Layer {i}")
        print([neuron[:-1] for neuron in layer])  # input weights
        print([neuron[-1] for neuron in layer])  # biases


# Model 1: Regression (ReLU -> Linear -> Linear, MSE)
print("------ Model 1 ------")
print("------ Task 1 ------")
nn = build_regression_network()
inputs = [1.5, 0.5]
expects = [0.8, 1]
loss_fn = MSE()
outputs = nn.forward(inputs)
nn.backward(loss_fn.get_output_gradients(outputs, expects))
nn.zero_grad(0.01)
print_new_weights(nn)

print("------ Task 2 ------")
nn = build_regression_network()
for epoch in range(1000):
    outputs = nn.forward(inputs)
    nn.backward(loss_fn.get_output_gradients(outputs, expects))
    nn.zero_grad(0.01)
print("Loss", loss_fn.get_loss(nn.forward(inputs), expects))


# Model 2: Binary Classification (ReLU -> Sigmoid, BCE)
print("\n------ Model 2 ------")
print("------ Task 1 ------")
nn = build_binary_network()
inputs = [0.75, 1.25]
expects = [1]
loss_fn = BinaryCrossEntropy()
outputs = nn.forward(inputs)
nn.backward(loss_fn.get_output_gradients(outputs, expects))
nn.zero_grad(0.1)
print_new_weights(nn)

print("------ Task 2 ------")
nn = build_binary_network()
for epoch in range(1000):
    outputs = nn.forward(inputs)
    nn.backward(loss_fn.get_output_gradients(outputs, expects))
    nn.zero_grad(0.1)
print("Loss", loss_fn.get_loss(nn.forward(inputs), expects))
