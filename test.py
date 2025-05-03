from dataset import CSVDataset
from model.serialization import load_model
from utils import argmax

dataset = CSVDataset('src/test.csv')
layer_sizes = [784, 128, 64, 10]
net = load_model('model.json', layer_sizes)

correct = 0
for d, l in zip(dataset.images, dataset.labels):
    pred = net.predict(d)
    if pred == l:
        correct += 1

print(f'Accuracy: {(correct / len(dataset) * 100):.2f}%')
