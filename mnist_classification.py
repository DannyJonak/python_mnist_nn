
from keras.datasets import mnist
import feed_forward_net as ff
import resources.nn_functions as nnf
import numpy as np

def one_hot(L, dim) -> list:
    """
    L is a list of integers taking values from 0 to dim - 1
    returns one hot representation of elements of L
    """

    new_L = np.empty((len(L), dim))
    for i in range(len(L)):
        temp = np.zeros(dim)
        temp[L[i]] = 1
        new_L[i] = temp
    return new_L


def flatten(X):
    """
    returns X with elements of X flattened
    """

    return X.reshape(np.shape(X)[0], np.shape(X)[1] * np.shape(X)[2])


#test feed_forward_net on mnist digits
def main():

    #load mnist data
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    #prepare data
    #flatten input data
    val_X, val_y = flatten(train_X[10000: 20000]) / 255, one_hot(train_y[10000:20000], 10)
    train_X, test_X = flatten(train_X[:10000]) / 255, flatten(test_X) / 255

    #use one hot encoding for training labels
    train_y, test_y = one_hot(train_y[:10000], 10), one_hot(test_y, 10)

    #build net
    net = ff.FFnet(nnf.relu, nnf.relu_grad)

    net.add_layer(784, 150)
    net.add_layer(150, 10, True)

    #train on train_X, train_y
    print("Beginning Training!")
    net.train(train_X, train_y, 5)

    #compute accuracy on a validation set
    #get predictions
    train_labels = []
    for x in val_X:
        train_labels.append(np.argmax(net.predict(x)))
    
    #count correct predictions
    correct = 0
    for i in range(len(val_y)):
        if np.argmax(val_y[i]) == train_labels[i]:
            correct += 1
    
    #compute accuracy
    print(correct/len(val_y))


if __name__=="__main__":

    main()


