from AtomicTensor.Tensor import Tensor


class Variable(object):
    def __init__(self, data, precedents=None, operator=None, requires_grad=True):
        assert isinstance(data,Tensor)
        self.data = data
        self.precedents = precedents
        self.operator = operator
        self.requires_grad = requires_grad
        if requires_grad:
            self.grad =Tensor().full(self.data.shape(),0)
