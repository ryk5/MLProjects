import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes, activation='sigmoid'):
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.weights = [np.random.randn(y, x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]   
        self.biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]
        self.activation = activation

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def forward_propagation(self, X):
        activations = [X]
        zs = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activations[-1]) + b
            activations.append(self.sigmoid(z))
            zs.append(z)
        return activations, zs

    def compute_cost(self, predicted, y):
        m = y.shape[1]
        return -1 / m * np.sum(y * np.log(predicted))

    def backward_propagation(self, X, y, activations, zs):
        m = y.shape[1]
        delta = activations[-1] - y
        gradients_w = []
        gradients_b = []
        for i in range(self.num_layers - 2, -1, -1):
            delta = np.dot(self.weights[i].T, delta) * self.sigmoid_derivative(zs[i])
            gradient_w = np.dot(delta, activations[i].T) / m
            gradient_b = np.mean(delta, axis=1, keepdims=True)
            gradients_w.insert(0, gradient_w)
            gradients_b.insert(0, gradient_b)
        return gradients_w, gradients_b

    def update_parameters(self, gradients_w, gradients_b, learning_rate):
        for i in range(self.num_layers - 1):
            self.weights[i] -= learning_rate * gradients_w[i]
            self.biases[i] -= learning_rate * gradients_b[i]

    def train(self, X, y, epochs=1000, learning_rate=0.01):
        for epoch in range(epochs):
            activations, zs = self.forward_propagation(X)
            cost = self.compute_cost(activations[-1], y)
            gradients_w, gradients_b = self.backward_propagation(X, y, activations, zs)
            self.update_parameters(gradients_w, gradients_b, learning_rate)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Cost: {cost}")

    def predict(self, X):
        activations, _ = self.forward_propagation(X)
        return activations[-1]

if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T
    y = np.array([[0, 1, 1, 0]])

    nn = NeuralNetwork(layer_sizes=[2, 3, 1])
    nn.train(X, y, epochs=5000, learning_rate=0.1)

    predictions = nn.predict(X)
    print("Predictions:", predictions)
