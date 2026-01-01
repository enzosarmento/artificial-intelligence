from typing import List, Callable
import numpy as np


class Neuron:

    def __init__(self, weights: List[float], bias: float):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs: List[float], activation_func: Callable[[float], float]) -> float:
        total = np.dot(inputs, self.weights) + self.bias
        return activation_func(total)
