from AtomicTensorCpp.Layer import Layer




class Relu(Layer):
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


class Sigmoid(Layer):
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