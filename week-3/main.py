class Network:
    def __init__(self, layers):
        self.layers = layers

    def neuron_output(self, weights, inputs):
        # The last weight is for the bias (whose input is always 1).
        input_weights = weights[:-1]
        bias_weight = weights[-1]
        weighted_sum = sum(w * x for w, x in zip(input_weights, inputs))
        return weighted_sum + bias_weight

    def forward(self, inputs):
        values = inputs
        for layer in self.layers:
            values = [self.neuron_output(neuron, values) for neuron in layer]
        return values


# Network 1
network_1 = Network(
    [
        # hidden layer: weights are [w_X1, w_X2, w_bias]
        [
            [0.5, 0.2, 0.3],  # hidden neuron 1
            [0.6, -0.6, 0.25],  # hidden neuron 2
        ],
        # output layer: weights are [w_hidden1, w_hidden2, w_bias]
        [
            [0.8, 0.4, -0.5],  # O1
        ],
    ]
)

print("Network 1")
print(network_1.forward([1.5, 0.5]))
print(network_1.forward([0, 1]))


# Network 2
network_2 = Network(
    [
        # hidden layer 1: [w_X1, w_X2, w_bias]
        [
            [0.5, 1.5, 0.3],  # neuron 1
            [0.6, -0.8, 1.25],  # neuron 2
        ],
        # hidden layer 2: [w_n1, w_n2, w_bias]
        [
            [0.6, -0.8, 0.3],  # neuron 3
        ],
        # output layer: [w_n3, w_bias]
        [
            [0.5, 0.2],  # O1
            [-0.4, 0.5],  # O2
        ],
    ]
)

print("\nNetwork 2")
print(network_2.forward([0.75, 1.25]))
print(network_2.forward([-1, 0.5]))
