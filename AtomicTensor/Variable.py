from .Tensor import *

def make_variable_like(source, target):    #scalar (128,1)
    '''Make the operand a tensor.'''
    if isinstance(source, Variable):
        return source
    assert isinstance(source, int) or isinstance(source, float), 'Cannot convert to tensor'
    # return Tensor().full(target.shape, source)
    # s = (1,) * len(target.data.shape)  #tumple (1,1)
    # a = np.zeros(s, dtype=np.float32) ()  #ndarray  (1,1)
    # b = a + source
    b=make_tensor_like(source,target.data)
    return Variable(b,requires_grad=False)

def attach_grad(u, grad):
    '''
    During backpropagation, attach computed grad into precedents.
    If forward process includes broadcasting, unbroadcast grad.
    '''
    if not u.requires_grad:
        return

    if u.grad.shape == grad.shape:
        u.grad += grad
        return

    # unbroadcasting
    for dim, chn in enumerate(u.grad.shape):
        if chn != grad.shape[dim]:
            assert chn == 1, "Backward unbroadcasting errors"
            grad = grad.mean(axis=dim, keepdims=True)

    u.grad += grad


class Variable:
    # def __init__(self, data, shape=None, usage='parameter', precedents=None, operator=None, autograd=True, name=None):
    def __init__(self, data, precedents=None, operator=None, requires_grad=True):
        self.data=Tensor(data)

        if requires_grad:
            self.grad = Tensor.full(self.data.shape, 0)
        else:
            self.grad = Tensor.empty_like(self.data)

        self.precedents = precedents
        self.operator = operator
        self.requires_grad = requires_grad
        if precedents:
            self.leaf = False
        else:
            self.leaf = True

    def __neg__(self):
        return Variable(-self.data, precedents=[self], operator='neg')

    def __pos__(self):
        return self

    def abs(self):
        return Variable(self.data.abs(), precedents=[self], operator='abs')

    def sum(self, dim=None):
        return Variable(self.data.sum(dim), precedents=[self], operator='sum')

    def mean(self, dim=None):
        return Variable(self.data.mean(dim), precedents=[self], operator="mean")

    def exp(self):
        return Variable(self.data.exp(), precedents=[self], operator='exp')

    def log(self):
        return Variable(self.data.log(), precedents=[self], operator='log')

    def __add__(self, X):
        X = make_variable_like(X, self)
        return Variable(self.data + X.data, precedents=[self, X], operator='add')

    def __radd__(self, X):
        return Variable.__add__(self, X)

    def __sub__(self, X):
        X = make_variable_like(X, self)
        return Variable(self.data - X.data, precedents=[self, X], operator='-')

    def __rsub__(self, X):
        X = make_variable_like(X, self)
        return Variable.__sub__(X, self)

    def __pow__(self, X):
        X = make_variable_like(X, self)
        X.requires_grad = False
        return Variable(self.data ** X.data, precedents=[self, X], operator='**')

    def __truediv__(self, X):
        assert isinstance(X, int) or isinstance(            X, float),         'Right divide only supports int or float. Please use *'
        X = make_variable_like(X, self)
        X.data = 1.0 / X.data
        return Variable.__mul__(self, X)

    def __rtruediv__(self, X):
        assert isinstance(X, int) or isinstance(            X, float),         'Right divide only supports int or float. Please use *'
        X = make_variable_like(X, self)
        return Variable(X.data / self.data, precedents=[self, X], operator='/')

    def __mul__(self, X):
        X = make_variable_like(X, self)
        return Variable(self.data * X.data, precedents=[self, X], operator='*')

    def __rmul__(self, X):
        return Variable.__mul__(self, X)

    def __matmul__(self, X):
        return Variable(self.data @ X.data, precedents=[self, X], operator='@')

    def __gt__(self, X):
        X = make_variable_like(X, self)
        return Variable(self.data > X.data, precedents=[self, X], operator='>', requires_grad=False)

    def __getitem__(self, slic):
        return Variable(self.data[slic], precedents=[self, slic], operator='slice')

    def view(self, *shape):
        return Variable(self.data.reshape(*shape))

    def __repr__(self):
        return "Tensor({})".format(self.data)

    @staticmethod
    def zeros(args):
        return Variable(Tensor().zeros(args))

    @staticmethod
    def randn(args):
        return Variable(Tensor().randn(*args))

    @staticmethod
    def ones(args):
        return Variable(Tensor.ones(args))

    def backward(self, internal=False):
        if not internal:
            self.grad = Tensor().ones_like(self.grad)

        if self.leaf:
            return

        from ctensor.Operator import Operator
        if isinstance(self.operator, Operator):
            self.operator.backward(self, self.precedents)

        elif self.operator == 'neg':
            u, = self.precedents
            u.grad += -self.grad

        elif self.operator == 'abs':
            u, = self.precedents
            u.grad += self.grad * Tensor().sign(u.data)

        elif self.operator == 'exp':
            u, = self.precedents
            u.grad += self.grad * self.data

        elif self.operator == 'log':
            u, = self.precedents
            u.grad += self.grad * (1.0 / u.data)

        elif self.operator == 'sum':
            u, = self.precedents
            u.grad += self.grad

        elif self.operator == 'mean':
            u, = self.precedents
            elements = 1
            for s in u.grad.shape:
                elements *= s
            u.grad += self.grad / elements

        elif self.operator == 'slice':
            u, slic = self.precedents
            u.grad[slic] += self.grad
            # c=0  # ignore warning

        elif self.operator == '+':
            u, v = self.precedents
            attach_grad(u, self.grad)
            attach_grad(v, self.grad)

        elif self.operator == '-':
            u, v = self.precedents
            attach_grad(u, self.grad)
            attach_grad(v, -self.grad)

        elif self.operator == '/':
            u, v = self.precedents
            attach_grad(u, -self.grad * v.data / (u.data ** 2))

        elif self.operator == '*':
            u, v = self.precedents
            attach_grad(u, self.grad * v.data)
            attach_grad(v, self.grad * u.data)

        elif self.operator == '**':
            u, v = self.precedents
            attach_grad(u, self.grad * u.data ** (v.data - 1) * v.data)

        elif self.operator == '>':
            u, v = self.precedents
            attach_grad(u, self.grad * (u.data > v.data))

        elif self.operator == '@':
            u, v = self.precedents
            if len(self.data.shape) == 3:
                attach_grad(u, self.grad @ (v.data.T3D()))
                attach_grad(v, u.data.T3D() @ self.grad)
            else:
                attach_grad(u, self.grad @ v.data.T)
                attach_grad(v, u.data.T @ self.grad)

        for p in self.precedents:
            if isinstance(p, Variable) and p.requires_grad:
                p.backward(internal=True)
