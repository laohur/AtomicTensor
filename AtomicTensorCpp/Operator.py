class Operator(object):
    def __init__(self, name):
        self.name = name

    def forward(self):
        pass

    def backward(self):
        pass

    def __call__(self, *x):
        return self.forward(*x)


class Add(Operator):
    def __init__(self, name="add"):
        self.name = name

    def forward(self, a, b):
        from .Tensor import Tensor
        re = Tensor(a.data + b.data)
        return re

    def backward(self, a):
        from .Tensor import Tensor
        grad = Tensor(a.grad.data)
        a.parents[0].backward(grad, a)
        a.parents[1].backward(grad, a)
