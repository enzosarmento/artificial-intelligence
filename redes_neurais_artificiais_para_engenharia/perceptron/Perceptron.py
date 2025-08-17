import numpy as np

class Perceptron:
    def __init__(self, weights, activation_func):
        self.weights = weights  # O vetor de pesos terá como primeiro peso a bias.
        self.activation_func = activation_func
        self.epochs = 0

    def feedforward(self, inputs):
        return sum(self.weights * inputs)

    def predict(self, inputs):
        u = self.feedforward(inputs)
        return self.activation_func(u)

    def train(self, data, learning_rate=0.01, max_epochs=1000):
        print(f"Peso inicial: {self.weights}")

        for epoch in range(max_epochs):
            error_found_in_epoch = False

            for _, row in data.iterrows():
                x1, x2, x3, d = row
                x = np.array([-1, x1, x2, x3])
                y = self.predict(x)

                if y != d:
                    self.weights = self.weights + learning_rate * (d - y) * x
                    error_found_in_epoch = True

            if not error_found_in_epoch:
                print(f"Treinamento concluído na época: {epoch + 1}")
                self.epochs = epoch + 1
                break

        print(f"Peso final: {self.weights}")