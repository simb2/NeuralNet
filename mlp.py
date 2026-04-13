from layer import Layer
from neuron import Neuron, sigmoid
import numpy as np

def binary_cross_entropy(predicted: float, actual: float) -> float:
    """Binary cross entropy — the loss which will be used to train the model.
    :param predicted: the predicted outcome for an observation.
    :param actual: the actual outcome for an observation.
    """
    predicted = np.clip(predicted, 1e-7, 1 - 1e-7)
    return -(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted))

def bce_derivative(predicted: float, actual: float) -> float:
    """Derivative of binary cross entropy. Needed for back propagation.
    :param predicted: the predicted outcome for an observation.
    :param actual: the actual outcome for an observation.
    """
    predicted = np.clip(predicted, 1e-7, 1 - 1e-7)
    return (predicted - actual) / (predicted * (1 - predicted))


class MLP:
    """A multilayer perceptron.
    :param list_num_neurons: a list containing the number of neurons in each hidden layer.
    :param input: an example input array, used to infer input size.
    """

    def __init__(self, list_num_neurons: list[int], input: np.ndarray):
        self.list_num_neurons = list_num_neurons
        self.layers = [None] * len(list_num_neurons)

        for i, num_neurons in enumerate(list_num_neurons):
            len_inputs = input.size if i == 0 else list_num_neurons[i - 1]
            self.layers[i] = Layer(num_neurons=num_neurons, len_inputs=len_inputs)

        # Output neuron: takes output of last hidden layer as input
        self.output_neuron = Neuron(len_inputs=list_num_neurons[-1])

    def forward(self, input: np.ndarray) -> float:
        """Run a forward pass through hidden layers then output neuron.
        :param input: a single observation as a 1D numpy array.
        :return: predicted probability between 0 and 1.
        """
        last_out = input
        for layer in self.layers:
            last_out = layer.activate(last_out)
        return self.output_neuron.output(last_out)

    def backpropagation(self, learning_rate: float, input: np.ndarray, actual: float):
        """Run forward then backward pass, updating parameters along the way.
        :param learning_rate: step size for gradient descent.
        :param input: a single observation as a 1D numpy array.
        :param actual: the true 0/1 label for this observation.
        """
        # Forward pass
        predicted = self.forward(input)

        # BCE derivative
        bce_grad = bce_derivative(predicted, actual)

        # Sigmoid derivative at output neuron
        sig_grad = predicted * (1 - predicted)

        # Combined scalar gradient at output neuron
        delta = bce_grad * sig_grad

        # Update output neuron parameters
        self.output_neuron.weights -= learning_rate * delta * self.output_neuron.prev_input
        self.output_neuron.intercept -= learning_rate * delta

        # Jacobian to pass back into hidden layers
        next_jacobi = delta * self.output_neuron.weights

        # Backward pass through hidden layers
        for layer in reversed(self.layers):
            next_jacobi = layer.backpropagate(learning_rate, next_jacobi)

    def train(self, X: np.ndarray, y: np.ndarray, learning_rate: float, epochs: int):
        """Train the MLP using SGD.
        :param X: 2D array of observations, rows are observations.
        :param y: 1D array of binary labels.
        :param learning_rate: step size for gradient descent.
        :param epochs: number of full passes over the dataset.
        """
        for epoch in range(epochs):
            total_loss = 0.0
            for i in range(len(X)):
                predicted = self.forward(X[i])
                total_loss += binary_cross_entropy(predicted, y[i])
                self.backpropagation(learning_rate, X[i], y[i])

            avg_loss = total_loss / len(X)
            print(f"Epoch {epoch + 1}/{epochs} — Loss: {avg_loss:.4f}")

    def output_layer(self, input: np.ndarray) -> float:
        """Run just the output neuron on the given input and return a probability.
        :param input: output of the last hidden layer as a 1D numpy array.
        :return: predicted probability between 0 and 1.
        """
        return self.output_neuron.output(input)