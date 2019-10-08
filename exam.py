import numpy as np
import tqdm
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from AtomicTensor.Activition import Relu
from AtomicTensor.Layer import Layer, Sequence, Linear
from AtomicTensor.Loss import MSE
from AtomicTensor.Optimizer import SGD


class Mynet(Layer):
    def __init__(self):
        super(Mynet, self).__init__()
        self.criterion = MSE()
        self.layers = Sequence(
            Linear(2, 100),
            Relu(),
            Linear(100, 1)
        )
        self.optimizer = SGD(self.parameters(), lr=0.0001)

    def parameters(self):
        return self.layers.parameters()

    def forward(self, *x):
        x = x[0]
        return self.layers.forward(x)

    def backward(self, grad=None):
        grad = self.criterion.backward(grad)
        self.layers.backward(grad)

    def fit_one(self, input, label, train=True):
        self.optimizer.zero_grad()
        pred = self.forward(input)
        loss = -1
        if train:
            loss = self.criterion(pred, label)
            self.backward()
            self.optimizer.step()
        return loss, pred


if __name__ == "__main__":
    mynet = Mynet()
    x = np.linspace(-20, 20, 41)
    y = np.linspace(-20, 20, 41)
    X, Y = np.meshgrid(x, y)
    t = np.dstack((X, Y))
    t = t.reshape(-1, 2)
    label = t[:, 0] + t[:, 1]
    h = label.reshape(-1, 1)

    for i in tqdm.tqdm(range(1000)):
        running_loss = 0.0
        for row in range(t.shape[0]):
            input = t[row:row + 1]
            label = h[row:row + 1]
            loss, pred = mynet.fit_one(input, label)
            running_loss += loss
        if i % 100 == 0:
            valpred = []
            print(" loss: ", running_loss / row)
            for row in range(t.shape[0]):
                input = t[row:row + 1]
                loss, pred = mynet.fit_one(input, label, train=False)
                valpred.append(pred)
            valpred = np.array(valpred)
            valpred = valpred.reshape(41, 41)
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.plot_surface(X, Y, valpred, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
            plt.show()
