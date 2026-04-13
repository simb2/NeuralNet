from neuron import Neuron, sigmoid
import numpy as np
import math

def sigmoid_diff_chain(input: np.ndarray, weights: np.ndarray, intercept: np.ndarray):
    """Used in computing the partial derivatives of the sigmoid function with respect to the parameters.
    :param input: The input from the previous layer.
    :param weights: The weights for this function.
    :param intercept: The intercept for this function.
    """
    term1 = math.exp(-1 * (np.dot(weights, input) + intercept))
    return term1 / ((1 + term1) ** 2)

class Layer:
    """A hidden layer of neurons.
    :param num_neurons: The number of neurons in this layer.
    :param len_inputs: The length of the input vector.
    """

    def __init__(self, num_neurons: int, len_inputs: int):
        self.neurons = [Neuron(len_inputs) for _ in range(num_neurons)]
        self.prev_input = None

    def activate(self, inputs: np.ndarray) -> np.ndarray:
        """Gathers the output of each neuron and sends it to the next layer."""
        out = np.empty(len(self.neurons))
        self.prev_input = inputs
        for i, neuron in enumerate(self.neurons):
            out[i] = neuron.output(input=inputs)
        return out

    def backpropagate(self, learning_rate: float, next_jacobi: np.ndarray):
        """Returns the jacobian of this layer w.r.t the previous input.
        Also updates the weights and intercepts of the neurons.
        :param learning_rate: The learning rate for this step.
        :param next_jacobi: The jacobian from the next layer.
        """
        jacobian = np.empty((len(self.neurons), len(self.prev_input)))

        for i, neur in enumerate(self.neurons):
            diff = sigmoid_diff_chain(input=neur.prev_input,
                                      weights=neur.weights,
                                      intercept=neur.intercept)
            jacobian[i, :] = diff * neur.weights

            neur.weights -= learning_rate * next_jacobi[i] * diff * neur.prev_input
            neur.intercept -= learning_rate * next_jacobi[i] * diff

        return next_jacobi @ jacobian