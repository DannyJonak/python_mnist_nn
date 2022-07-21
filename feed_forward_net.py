#feed forward neural network module

import numpy as np
import resources.nn_layer as l
import resources.customexceptions as cs

class FFnet:
    """
    Class to represent a feed forward neural network with softmax output and cross entropy loss.
    Made to classify mnist digits.
    """


    def __init__(self, activation, activation_grad):

        self.layers = []
        self.activation = activation
        self.activation_grad = activation_grad


    def add_layer(self, input_size, num_nodes, output_layer=False):
        
        if self.layers:
            if self.layers[-1].kind == 'output':
                raise cs.OutputLayerError("Unable to add a new layer to an output layer.")
            if self.layers[-1].num_nodes != input_size:
                raise cs.LayerSizeError("The input size of this layer does not match the output size of the previous layer.")

        if not output_layer:
            new_layer = l.HiddenLayer(input_size, num_nodes, self.activation, self.activation_grad)
        else:
            new_layer = l.OutputLayer(input_size, num_nodes)
        self.layers.append(new_layer)


    def remove_layer(self):

        if not self.layers:
            return
        self.layers.pop()

    
    def predict(self, sample):

        data = sample
        #iterate through layers and compute output for each layer
        #output from previous layer is input for next layer
        for layer in self.layers:

            data = layer.forward_pass(data)
        
        return data


    def batch_predict(self, samples):
        
        results = []

        for sample in samples:
            results.append(self.predict(sample))
        
        return results

    
    def train(self, input_data, target, epochs, learning_rate=0.01, momentum=0.9):
        
        #return if self.layers is empty and check that self.layers includes an output layer
        if not self.layers:
            return
        if self.layers[-1].kind != "output":
            raise cs.OutputLayerError("Net cannot be trained without an output layer.")

        #implement gradient descent with momentum
        for epoch in range(epochs):
            
            loss = 0

            for i in range(len(input_data)):
                
                cur_data = input_data[i]
                cur_target = target[i]

                #forward pass through net and compute outputs
                for layer in self.layers:
                    
                    cur_data = layer.forward_pass(cur_data)

                #update loss
                loss -= np.sum(np.log(cur_data) * cur_target)
                
                #backpropegation
                #update output layer first

                #iterate through hidden layers and update weights, bias, and errors for each

                #input for get_grads of the output layer is target hence we set cur_error to cur_target for first iteration
                cur_error = cur_target
                for j in range(len(self.layers) - 1, -1, -1):
                    
                    cur_layer = self.layers[j]
                    cur_error, weight_grad, bias_grad = cur_layer.get_grads(cur_error)

                    #compute gradients with momentum
                    weight_update = (1 - momentum) * weight_grad + (momentum) * cur_layer.weight_chng
                    bias_update = (1 - momentum) * bias_grad + (momentum) * cur_layer.bias_chng

                    #update current weights and bias
                    cur_layer.weights = cur_layer.weights - (learning_rate) * weight_update
                    cur_layer.bias = cur_layer.bias - (learning_rate) * bias_update

                    #store changes used in weight and bias updates for momentum in next iteration
                    cur_layer.weight_chng, cur_layer.bias_chng = weight_update, bias_update

            print("Epoch: ", epoch, " Loss: ", loss/len(input_data))
