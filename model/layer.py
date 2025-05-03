import random
import math

from typing import Literal

class Neuron:
    def __init__(
            self,
            num_inputs: list[float],
            activation: Literal['relu', 'softmax'] = 'relu',
            dropout: float = 0.0
    ) -> None:
        self.weights: list[float] = [random.uniform(-0.2, 0.2) for _ in range(num_inputs)] # список весов
        self.bias: float = 0.0                                      # смещение
        self.inputs: list[float] = []                               # входные значения нейрона
        self.output: float = 0.0                                    # выходное значение нейрона
        self.delta: float = 0.0                                     # значение ошибки для корректировки весов
        self.activation: Literal['relu', 'softmax'] = activation    # функция активации
        self.dropout: float = dropout                               # вероятность отключения нейрона при тренировки
        self.dropped: bool = False                                  # показатель "выключенности" нейрона

    def activate(self, inputs: list[float], train: bool = True) -> float:
        """Вычисление выходного значения нейрона"""
        self.inputs = inputs
        z = sum(w * i for w, i in zip(self.weights, inputs)) + self.bias # сумма взвешеных входов со смещением
        if train and self.dropout and random.random() <= self.dropout:
            self.output = 0.0
            self.dropped = True
            return self.output
        if self.activation == 'relu':
            self.output = max(0.01 * z, z)
        elif self.activation == 'softmax':
            self.output = z
        return self.output

    def derivative(self) -> float:
        """Нормализация выходного значения"""
        if self.activation == 'relu':
            return 1.0 if self.output > 0 else 0.0
        return 1.0


class Layer:
    def __init__(
            self,
            num_neurons: int,               # количество входов
            num_inputs_per_neuron: int,     # количество выходов
            activation: str = 'relu',
            dropout: float = 0.0
    ) -> None:
        self.neurons: list[Neuron] = [Neuron(num_inputs_per_neuron, activation, dropout) for _ in range(num_neurons)]
        self.activation: Literal['relu', 'softmax'] = activation

    def forward(self, inputs: list[float], train: bool = True) -> list[float]:
        """Обработка каждого входа"""
        outputs = [neuron.activate(inputs, train=train) for neuron in self.neurons]
        if self.activation == 'softmax':
            """Преобразуем все выходы в положительные, что в сумме дают 1"""
            # max_logit = max(n.output for n in self.neurons)
            # exps = [math.exp(n.output - max_logit) for n in self.neurons]
            max_output = max(neuron.output for neuron in self.neurons)  # находим максимальное значение
            exps = [math.exp(neuron.output - max_output) for neuron in self.neurons]
            total = sum(exps)
            for i, neuron in enumerate(self.neurons):
                neuron.output = exps[i] / total
            outputs = [neuron.output for neuron in self.neurons]
        return outputs
