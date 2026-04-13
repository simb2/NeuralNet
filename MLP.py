from Layer import Layer
import numpy as np
class MLP:
    """A multilayer preceptron. 
    
    :param list_num_neurons: a list containing the number of neurons in each layer. The initial
    layer is denoted by 1.
    :param num_layers: The number of layers in the MLP.
    """
    
    def __init__(self, list_num_neurons: list[int], input: np.ndarray):
        """
        :param list_num_neurons: a list containing the number of neurons in each layer. The initial
        layer is denoted by 1.
        :param num_layers: The number of layers in the MLP.
        """
        self.list_num_neurons = list_num_neurons
        self.layers = []*len(list_num_neurons)
        self.input_size = input.size
        i = 1
        for i, num_neurons in enumerate(list_num_neurons):
            len_inputs = input.size if i == 0 else self.layers[i - 1]
            self.layers[i] = Layer(num_neurons=num_neurons, len_inputs=len_inputs)
    
    def backpropogation:
        pass

    def train(self):
        pass