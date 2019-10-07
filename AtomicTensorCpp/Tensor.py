import numpy as np

from AtomicTensorCpp.Operator import Add


class Tensor(object):
    def __init__(self, data):
        self.data = np.array(data)

    def __add__(self, other):
        re = Add(self, other)
        return re


class Variable(object):
    def __init__(self, weight, gradw, bias, gradb):
        self.weight = weight
        self.gradw = gradw
        self.bias = bias
        self.gradb = gradb
        self.parent = []
        self.operator = None
        self.autograd = True
        self.vt = np.zeros(self.weight.shape)  # old grad
