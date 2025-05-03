from model.layer import Layer
from utils import one_hot, argmax
import math


class Network:
    def __init__(
            self,
            layer_sizes: list[int],
            dropout: float = 0.0
    ) -> None:
        """Создание слоёв, где выходным всегда будет softmax"""
        self.layers: list[Layer] = []
        for i in range(1, len(layer_sizes)):
            activation = 'softmax' if i == len(layer_sizes) - 1 else 'relu'
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i-1], activation=activation, dropout=dropout))

    def forward(self, inputs: list[float], train: bool = True) -> list[float]:
        """Последовательный пропуск входных данных через каждый слой"""
        for layer in self.layers:
            inputs = layer.forward(inputs, train)
        return inputs

    def backward(self, expected: list[float], learning_rate: float) -> None:
        """Обратно распространение ошибки(backpropagation)"""
        # Выходной слой
        last = self.layers[-1]
        for i, neuron in enumerate(last.neurons):
            neuron.delta = neuron.output - expected[i]

        # Скрытые слои
        for i in reversed(range(len(self.layers) - 1)):
            current = self.layers[i]
            next_layer = self.layers[i + 1]
            for i, neuron in enumerate(current.neurons):
                if neuron.dropped:
                    continue
                error = sum(n.weights[i] * n.delta for n in next_layer.neurons)
                neuron.delta = error * neuron.derivative()

        # Обновление весов
        for layer in self.layers:
            for neuron in layer.neurons:
                if neuron in layer.neurons:
                    if neuron.dropped:
                        continue
                    for j in range(len(neuron.weights)):
                        neuron.weights[j] -= learning_rate * neuron.delta * neuron.inputs[j]
                    neuron.bias -= learning_rate * neuron.delta

    def train(self, data: list[list[int | float]], labels: list[int], epochs: int, learning_rate: float) -> None:
        """Обучение модели"""
        for epoch in range(epochs):
            total_loss = 0.0
            for d, l in zip(data, labels):
                for layer in self.layers:
                    for neuron in layer.neurons:
                        neuron.dropped = False
                output = self.forward(d)
                expected = one_hot(l)
                for i, _ in enumerate(output):
                    output[i] = max(output[i], 1e-8)
                loss = -sum(expected[i] * math.log(output[i]) for i in range(len(expected)))

                total_loss += loss
                self.backward(expected, learning_rate)
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(data):.4f}')

    def predict(self, x):
        """Предсказание класса по выходному значению"""
        output = self.forward(x)
        return argmax(output)
