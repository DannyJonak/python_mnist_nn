# python_mnist_nn

Contains an implementation of a feed forward neural network with softmax output and cross entropy loss. <br />
The net is built using only python and numpy.

**nn_functions.py** contains activation functions and an implementation of the softmax function to be used in the neural network. <br />
**nn_layer.py** contains classes to implement the hidden and output layres of a feed forward neural network. <br />
**feed_forward_net.py** contains a class to implement a feed forward network. <br />
**mnist_classification.py** is a script that builds a network using the implementation found in feed_forward_net.py and then tests it on mnist digits. <br />

Required Libraries: <br />
Numpy <br />
Keras - to load mnist dataset <br />
