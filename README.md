# python_mnist_nn

Required Libraries:
Numpy
Keras - to load mnist dataset

Contains an implementation of a feed forward neural network with softmax output and cross entropy loss.\n
The net is built using only python and numpy.\n

nn_functions.py contains activation functions and an implementation of the softmax function to be used in the neural network.\n
nn_layer.py contains classes to implement the hidden and output layres of a feed forward neural network.\n
feed_forward_net.py contains a class to implement a feed forward network.\n
mnist_classification.py is a script that builds a network using the implementation found in feed_forward_net.py and then tests it on mnist digits.\n
