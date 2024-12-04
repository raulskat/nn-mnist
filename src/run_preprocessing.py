# run_preprocessing.py
import numpy as np
from data_preprocessing import load_data, preprocess_data

# Load and preprocess data
data = load_data('data/train.csv')
X_train, Y_train, X_dev, Y_dev = preprocess_data(data)

# Optionally print the shapes to verify
# print("Training set shape:", X_train.shape)
# print("Development set shape:", X_dev.shape)

# Save X_train and Y_train to files for later use in evaluate.py
np.save('data/X_train.npy', X_train)
np.save('data/Y_train.npy', Y_train)
np.save('data/X_dev.npy', X_dev)
np.save('data/Y_dev.npy', Y_dev)
