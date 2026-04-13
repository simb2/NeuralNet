import pytest
import numpy as np
import math

from neuron import Neuron, sigmoid

@pytest.fixture
def neuron_t():
    return Neuron(weights = np.array([1, 2, 3]), intercept = 3)

def test_neuron_output(neuron_t):
    assert neuron_t.output(input = np.zeros(3)) == sigmoid(3)
