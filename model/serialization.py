import json
from pathlib import Path
from model.network import Network

def save_model(network: Network, path: Path) -> None:
    """Выгрузка модели"""
    data = []
    for layer in network.layers:
        layer_data = []
        for neuron in layer.neurons:
            layer_data.append({
                'weights': neuron.weights,
                'bias': neuron.bias
            })
        data.append(layer_data)
    with open(path, 'w') as f:
        json.dump(data, f)


def load_model(path: Path, layer_sizes: list[float], dropout: float = 0.0) -> Network:
    """Загрузка модели"""
    net = Network(layer_sizes, dropout)
    with open(path, 'r') as f:
        data = json.load(f)
    for l, layer_data in enumerate(data):
        for n, neuron_data in enumerate(layer_data):
            net.layers[l].neurons[n].weights = neuron_data['weights']
            net.layers[l].neurons[n].bias = neuron_data['bias']
    return net
