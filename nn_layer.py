#neural net layer module

import numpy as np
import nn_functions as nnf

class Layer:
    """
    abstract class to represent a layer in a neural network
    """

    def __init__(self, input_size: int, num_nodes: int):

        self.weights = np.random.uniform(low = -0.5, high = 0.5, size = (input_size, num_nodes))
        self.bias = np.random.uniform(low = -0.5, high = 0.5, size = num_nodes)

        self.input = None
        self.output = None
        self.linear_output = None

        #next two attributes to be used to calculate momentum when training with gradient descent
        self.weight_chng = np.zeros(np.shape(self.weights))
        self.bias_chng = np.zeros(np.shape(self.bias))


    def forward_pass(self):
        raise NotImplementedError
    

    def get_grads(self):
        raise NotImplementedError


class HiddenLayer(Layer):
    """
    a class to represent a single layer in a neural net to be used in a neural net class
    """

    def __init__(self, input_size: int, num_nodes: int, activation, activation_grad):
    
        Layer.__init__(self, input_size, num_nodes)

        self.kind = "hidden"
        self.activation = activation
        self.activation_grad = activation_grad


    def forward_pass(self, data):

        self.input = data
        self.linear_output = np.dot(self.input, self.weights) + self.bias
        self.output = self.activation(self.linear_output)
        
        return self.output
    

    def get_grads(self, error):

        if self.input is not None and self.output is not None and self.linear_output is not None:
            
            bias_grad = error * self.activation_grad(self.linear_output)
            weights_grad = np.outer(self.input, bias_grad)
            input_error = np.dot(bias_grad, np.transpose(self.weights))

            return input_error, weights_grad, bias_grad


class OutputLayer(Layer):
    """
    class to represent the output layer of a nn with softmax output and cross entropy loss
    """

    def __init__(self, input_size: int, num_nodes: int):
        
        Layer.__init__(self, input_size, num_nodes)

        self.kind = "output"


    def forward_pass(self, data):
            
        self.input = data
        self.linear_output = np.dot(self.input, self.weights) + self.bias
        self.output = nnf.softmax(self.linear_output)

        return self.output


    def get_grads(self, label):

        bias_grad = self.output - label
        weights_grad = np.outer(self.input, bias_grad)
        input_error = np.dot(bias_grad, np.transpose(self.weights))

        return input_error, weights_grad, bias_grad
