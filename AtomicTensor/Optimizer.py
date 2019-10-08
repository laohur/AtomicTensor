from AtomicTensor.Layer import Layer


class SGD(Layer):
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