import numpy as np


# wait for mars xtensor
class Tensor(object):
    def __init__(self, data=None, backend="cpu"):
        self.backend = backend
        self.data = np.array(data)

    def full(self, size, value):
        return np.full(size, value)

    def __neg__(self):
        return Tensor(-self.data, backend=self.backend)

    def __add__(self, X):
        re = Tensor(self.data + X.data, backend=self.backend)
        return re

    def __pos__(self):
        return self

    def __sub__(self, X):
        return Tensor(self.data - X.data, backend=self.backend)

    def __pow__(self, X):
        return Tensor(self.data ** X.data)

    def __mul__(self, X):
        return Tensor(self.data * X.data)

    def __matmul__(self, X):
        return Tensor(self.data @ X.data)

    def __getitem__(self, slic):
        return Tensor(self.data[slice])

    def __truediv__(self, X):
        return Tensor(self.data * (1 / X.data))

    def __repr__(self):
        return "Tensor({})".format(self.data)

    def __str__(self):
        return str(self.data.__str__())

    def view(self, *shape):
        return Tensor(self.data.reshape(*shape))

    def abs(self):
        return Tensor(np.abs(self.data))

    def sum(self, dim=None):
        return Tensor(np.sum(self.data, dim))

    def mean(self, dim=None):
        return Tensor(np.mean(self.data, dim))

    def exp(self):
        return Tensor(np.exp(self.data))

    def log(self):
        return Tensor(np.log(self))

    def __ge__(self, X):
        return Tensor(self.data >= X.data)

    def __gt__(self, X):
        return Tensor(self.data > X.data)

    def transpose(self, *x):
        return Tensor(self.data.transpose(*x))

    def T(self):
        return Tensor(self.transpose(0, 2, 1))

    @staticmethod
    def zeros(args):
        return Tensor(np.zeros(args, dtype=np.float32))

    @staticmethod
    def randn(args):
        return Tensor(np.random.randn(*args, dtype=np.float32))

    @staticmethod
    def ones(args):
        return Tensor(np.ones(args, dtype=np.float32))
