#neural net functions module

import numpy as np

def tanh(x):

    return np.tanh(x)

def tanh_grad(x):

    return 1 - np.power((np.tanh(x)), 2)

def relu(x):

    return (x > 0) * x

def relu_grad(x):

    return (x > 0) * 1

def softmax(x):

    exp = np.exp(x - np.max(x))

    return exp / sum(exp)
