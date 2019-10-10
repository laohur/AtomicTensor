import numpy as np


def make_tensor_like(source, target):  # scalar (128,1)
    '''Make the operand a tensor.'''
    if isinstance(source, Tensor):
        return source
    assert isinstance(source, int) or isinstance(source, float), 'Cannot convert to tensor'
    # return Tensor().full(target.shape, source)
    s = (1,) * len(target.data.shape)  # tumple (1,1)
    a = np.zeros(s, dtype=np.float32)  # ndarray  (1,1)
    b = a + source
    c = np.zeros(s, dtype=np.float32) + source

    return Tensor(b)


# wait for mars xtensor
class Tensor(object):
    def __init__(self, data=None, dtype="float32", backend="cpu"):
        if isinstance(data, Tensor):
            self.data = data.data
        else:
            self.data = np.array(data, dtype=dtype)

        self.backend = backend
        self.dtype = self.data.dtype
        self.shape = self.data.shape
        self.size = self.data.size
        self.strides = self.data.strides

    # single operator
    def __neg__(self):
        return Tensor(-self.data, backend=self.backend)

    def __pos__(self):
        return self

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

    def __add__(self, X):
        X = make_tensor_like(X, self)
        return Tensor(self.data + X.data, backend=self.backend)

    def __radd__(self, X):
        return Tensor(X.data + self.data, backend=self.backend)

    def __sub__(self, X):
        x = make_tensor_like(X, self)
        return Tensor(self.data - X.data, backend=self.backend)

    def __rsub__(self, X):
        X = make_tensor_like(X, self)
        return Tensor(X.data - self.data)

    def __mul__(self, X):
        return Tensor(self.data * X.data)

    def __pow__(self, X):
        return Tensor(self.data ** X.data)

    def __matmul__(self, X):
        return Tensor(self.data @ X.data)

    def __truediv__(self, X):
        assert isinstance(X, int) or isinstance(X, float), 'Right divide only supports int or float. Please use *'
        # X = make_tensor_like(X, self)
        return Tensor(self.data / X)

    def __rtruediv__(self, X):
        X = make_tensor_like(X, self)
        return Tensor(X.data / self.data)

    def __getitem__(self, slic):
        return Tensor(self.data[slice])

    def __repr__(self):
        return "Tensor({})".format(self.data)

    def __str__(self):
        return str(self.data.__str__())

    def view(self, *shape):
        return Tensor(self.data.reshape(*shape))

    def min(self):
        return self.data.min

    def max(self):
        return self.data.max

    def __ge__(self, X):
        return Tensor(self.data >= X.data)

    def __gt__(self, X):
        X = make_tensor_like(X, self)
        return Tensor(self.data > X.data)

    def sign(self):
        return Tensor(np.sign(self.data))

    def transpose(self, *x):
        return Tensor(self.data.transpose(*x))

    def T(self):
        return Tensor(self.transpose(1, 0))

    def T3D(self):
        '''Transpose a 3-d matrix, [batch x N x M].'''
        return Tensor(self.data.transpose(0, 2, 1))

    @staticmethod
    def zeros(*args):
        return Tensor(np.zeros(*args))

    @staticmethod
    def randn(*args):
        return Tensor(np.random.randn(*args))

    @staticmethod
    def ones(args):
        return Tensor(np.ones(args))

    @staticmethod
    def ones_like(target):
        return Tensor(np.ones_like(target))

    @staticmethod
    def emppty_like(self):
        return Tensor(np.empty_like(self.data))

    @staticmethod
    def tensordot(*args):
        return Tensor(np.tensordot(*args))

    def numpy(self):
        return np.array(self.data)

    @staticmethod
    def full(size, value):
        return Tensor(np.full(size, value))


if __name__ == "__main__":
    a = np.random.randn(2, 3)
    b = np.random.randn(2, 3)
    c = np.random.randn(3, 5)
    X = Tensor(a)
    Y = Tensor(b)
    Z = Tensor(c)

    print(X + Y)
    print(X - Y)
    print(X * Y)
    W = X @ Z
    print(W.shape)

    U = Tensor().full(W.shape, 1)
    print(U)

    n=np.array([1])
    m=np.array(1)
    a=Tensor(m)
    b=158404
    c=a+b
    d=a/b
    e=0