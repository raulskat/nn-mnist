# src/train.py

import numpy as np
from model import init_params, forward_prop, backward_prop, update_params, get_predictions, get_accuracy
X_train = np.load('data/X_train.npy')
Y_train = np.load('data/Y_train.npy')
W1 = np.load('model/W1.npy')
b1 = np.load('model/b1.npy')
W2 = np.load('model/W2.npy')
b2 = np.load('model/b2.npy')
print(W1)
def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.99, 1000)
np.save('model/W1.npy', W1)
np.save('model/b1.npy', b1)
np.save('model/W2.npy', W2)
np.save('model/b2.npy', b2)