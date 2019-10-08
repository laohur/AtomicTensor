import numpy as np


from AtomicTensor.Tensor import Variable


class Layer(object):
    def __init__(self):
        pass

    def forward(self):
        raise NotImplemented

    def backward(self, grad):
        raise NotImplemented

    def __call__(self, *x):
        return self.forward(*x)

class Linear(Layer):
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

class Sequence(Layer):
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
