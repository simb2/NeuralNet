import numpy as np
import math

# should do a couple more like tanh and add a functionality that the
# neuron can be either a tanh function or a sigmoid function
def sigmoid(x: float):
    """
    :param x: a vector
    """
    return 1/(1 + math.exp(-1 * x))


# now to define a neuron

# what do we want a neuron to do? Well we want it to actually take an input, 
# transform to either be something close to one or zero, and then output that result right?



class Neuron:
    """Defines a neuron for a MLP. 
    :param weights: The weights for the pre-activation function
    :param b: The bias term for the pre-activation function
    """

    def __init__(self, len_inputs: int):
        self.weights = np.random.randn(len_inputs)
        self.intercept = np.random.randn(1)
        self.prev_input = None

    def output(self, 
        input: np.ndarray):
        """Output of this neuron, which is either fed forward to the
        next lair.
        
        :param inputs: a vector of inputs. 
        """
        self.prev_input = input
        return sigmoid(np.dot(self.weights, input) + self.intercept)


