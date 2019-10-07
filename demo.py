import numpy as np
import tqdm
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


class Variable(object):
    def __init__(self, weight, gradw, bias, gradb):
        self.weight = weight
        self.gradw = gradw
        self.bias = bias
        self.gradb = gradb
        # self.vt = np.zeros(self.weight.shape)  #old grad


class Network(object):
    def __init__(self):
        pass

    def forward(self):
        raise NotImplemented

    def backward(self, grad):
        raise NotImplemented

    def __call__(self, *x):
        return self.forward(*x)


class Linear(Network):
    def __init__(self, inplans, outplans):
        super(Linear, self).__init__()
        self.weight = np.random.randn(inplans, outplans) * 0.5
        self.bias = np.random.randn(outplans) * 0.5
        self.gradw = np.zeros(self.weight.shape)
        self.gradb = np.zeros(self.bias.shape)
        self.parameter = Variable(self.weight, self.gradw, self.bias, self.gradb)
        self.input = None

    def parameters(self):
        return self.parameter

    def forward(self, *x):
        x = x[0]
        self.input = x
        return np.dot(x, self.parameter.weight) + self.parameter.bias

    def backward(self, grad):
        self.gradb = grad
        self.gradw += np.dot(self.input.T, grad)
        return np.dot(grad, self.weight.T)


class Relu(Network):
    def __init__(self):
        super(Relu, self).__init__()
        self.input = None

    def forward(self, *x):
        x = x[0]
        self.input = x
        x[self.input <= 0] *= 0
        return x

    def backward(self, grad):
        grad[self.input > 0] = 1
        grad[self.input <= 0] = 0
        return grad


class Sigmoid(Network):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.input = None
        # self.eps=0

    def forward(self, *x):
        x = x[0]
        self.input = x
        x = 1 / (1 + np.exp(-x))
        return x

    def backward(self, grad):
        grad *= self.input(1 - self.input)
        return grad


class MSE(Network):
    def __init__(self):
        self.pred = None
        self.label = None
        self.loss = -1

    def forward(self, pred, label):
        self.pred = pred
        self.label = label
        self.loss = np.sum(np.square(pred - label) / 2.0)
        # size=pred.shape[0]
        # self.loss=np.sum(np.square(pred-label)/size/2,keepdims=True)
        return self.loss

    def backward(self, grad=None):
        self.grad = self.pred - self.label
        ret_grad = np.sum(self.grad, axis=0)
        return np.expand_dims(ret_grad, axis=0)

class SGD(Network):
    def __init__(self, parameters, lr=0.01, momentum=0.9):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.gradw *= 0
            parameter.gradb *= 0

    def step(self):
        for parameter in self.parameters:
            parameter.weight -= self.lr * parameter.gradw
            parameter.bias -= self.lr * parameter.gradb

class Sequence(Network):
    def __init__(self, *layers):
        super(Sequence, self).__init__()
        self.layers = []
        self.parameter = []
        for layer in layers:
            self.layers.append(layer)
        for layer in self.layers:
            if isinstance(layer, Linear):
                self.parameter.append(layer.parameters())

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, *x):
        x = x[0]
        for layer in self.layers:
            x = layer(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def parameters(self):
        return self.parameter  # !=parameters


class Mynet(Network):
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
