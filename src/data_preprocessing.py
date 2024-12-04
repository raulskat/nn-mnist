# src/data_preprocessing.py

import numpy as np
import pandas as pd

def load_data(file_path):
    data = pd.read_csv(file_path)
    data = np.array(data)
    return data

def preprocess_data(data):
    m, n = data.shape
    np.random.shuffle(data)  # shuffle before splitting into dev and training sets

    # Split the data into development and training sets
    data_dev = data[0:1000].T
    Y_dev = data_dev[0]
    X_dev = data_dev[1:n]
    X_dev = X_dev / 255.  # Normalize

    data_train = data[1000:m].T
    Y_train = data_train[0]
    X_train = data_train[1:n]
    X_train = X_train / 255.  # Normalize

    return X_train, Y_train, X_dev, Y_dev
