from typing import List, Callable, Tuple
import numpy as np
import pandas as pd

from utils.activation_functions import sigmoid


class NeuralNetwork:

    def __init__(self, layer_sizes: List[int]):
        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            output_size = layer_sizes[i + 1]
            weight_matrix = np.random.rand(output_size, input_size)
            self.weights.append(weight_matrix)
            bias_vector = np.random.rand(output_size, 1)
            self.biases.append(bias_vector)

    def feedforward_and_store(self, inputs, activation_func: Callable[[float], float]):
        activations = [inputs]
        potentials = []

        for w, b in zip(self.weights, self.biases):
            u = np.dot(w, activations[-1]) + b
            potentials.append(u)
            activations.append(activation_func(u))

        return activations, potentials

    def predict(self, x: np.ndarray) -> np.ndarray:
        activations, _ = self.feedforward_and_store(x, sigmoid)
        return activations[-1]

    def train(self, data: pd.DataFrame, learning_rate: float, max_epochs: int):
        prev_error = 1e10
        error_history = []
        for epoch in range(max_epochs):
            total_error = 0.0


            for _, row in data.iterrows():
                # Entrada (coluna) e saída esperada
                x = np.array(row[:-1]).reshape(-1, 1)
                d = np.array(row.iloc[-1]).reshape(-1, 1)

                # Feedforward
                activations, potentials = self.feedforward_and_store(x, sigmoid)

                # Cálculo do erro (MSE acumulado)
                y = activations[-1]
                total_error += np.sum((d - y) ** 2)

                # -------- Backpropagation --------
                deltas = []

                # Delta da camada de saída
                delta_output = (d - y) * y * (1 - y)
                deltas.append(delta_output)

                # Deltas das camadas escondidas (de trás pra frente)
                for l in range(len(self.weights) - 2, -1, -1):
                    a = activations[l + 1]
                    delta = np.dot(self.weights[l + 1].T, deltas[0]) * a * (1 - a)
                    deltas.insert(0, delta)

                # Atualização dos pesos e biases
                for l in range(len(self.weights)):
                    self.weights[l] += learning_rate * np.dot(deltas[l], activations[l].T)
                    self.biases[l] += learning_rate * deltas[l]

            print(f"Epoch {epoch + 1}/{max_epochs} - Erro: {total_error:.4f}")
            error_history.append(total_error)
            if abs(prev_error - total_error) < 1e-6:
                break
            prev_error = total_error

        return error_history