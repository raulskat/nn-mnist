# src/evaluate.py

import numpy as np
import matplotlib.pyplot as plt
from model import forward_prop, get_predictions
W1 = np.load('model/W1.npy')
b1 = np.load('model/b1.npy')
W2 = np.load('model/W2.npy')
b2 = np.load('model/b2.npy')
# Load X_train and Y_train from saved files
X_train = np.load('data/X_train.npy')
Y_train = np.load('data/Y_train.npy')
# X_dev = np.load('data/X_dev.npy')
# Y_dev = np.load('data/Y_dev.npy')
def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()
    
# test_prediction(0, W1, b1, W2, b2)
# test_prediction(1, W1, b1, W2, b2)
# test_prediction(2, W1, b1, W2, b2)
# test_prediction(3, W1, b1, W2, b2)

def visualize_misclassifications(X, Y, W1, b1, W2, b2):
    predictions = make_predictions(X, W1, b1, W2, b2)
    misclassified_indices = np.where(predictions != Y)[0]
    
    print(f"Number of Misclassifications: {len(misclassified_indices)}")
    
    for idx in misclassified_indices[:5]:  # Show up to 5 misclassified images
        current_image = X[:, idx].reshape((28, 28)) * 255
        plt.gray()
        plt.imshow(current_image, interpolation='nearest')
        plt.title(f"Prediction: {predictions[idx]}, Label: {Y[idx]}")
        plt.show()

# print("Visualizing Misclassifications on Training Set:")
# visualize_misclassifications(X_train, Y_train, W1, b1, W2, b2)


def evaluate_batch(X, Y, W1, b1, W2, b2):
    predictions = make_predictions(X, W1, b1, W2, b2)
    accuracy = np.mean(predictions == Y)
    print(f"Overall Accuracy: {accuracy:.2%}")
    return accuracy

# Evaluate the model on training data
# print("Evaluating Training Set:")
# evaluate_batch(X_train, Y_train, W1, b1, W2, b2)

# Optionally, load dev data and evaluate
# print("Evaluating Development Set:")
# evaluate_batch(X_dev, Y_dev, W1, b1, W2, b2)


