from AtomicTensorCpp.Layer import Layer
import numpy as np

class MSE(Layer):
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
