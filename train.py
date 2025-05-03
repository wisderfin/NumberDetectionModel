from dataset import CSVDataset
from model.network import Network
from model.serialization import save_model

dataset = CSVDataset('src/train.csv', 5000)

net = Network([784, 64, 32, 10], dropout=0.05) # изображение 28x28 (28*28=784)
net.train(dataset.images, dataset.labels, 10, 0.005)

save_model(net, 'model.json')
