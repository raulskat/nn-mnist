# Neural Network Model for Digit Classification

## Overview
This project implements a simple neural network for digit classification on the MNIST dataset.

## Requirements
- Python 3.x
- Numpy
- Pandas
- Matplotlib

## How to Run
1. Clone the repository.
2. Install the requirements:
    ```
    pip install -r requirements.txt
    ```
3. Preprocess the data:
    ```python
    from data_preprocessing import load_data, preprocess_data
    data = load_data('train.csv')
    X_train, Y_train, X_dev, Y_dev = preprocess_data(data)
    ```
4. Train the model:
    ```python
    from train import gradient_descent
    W1, b1, W2, b2 = gradient_descent(X_train, Y_train, alpha=0.99, iterations=1000)
    ```

5. Evaluate the model:
    ```python
    from evaluate import test_prediction
    test_prediction(0, X_train, Y_train, W1, b1, W2, b2)
    ```
## terminal

```powershell
python run_preprocessing.py
python train.py
python evaluate.py
```

## Results
The trained model can be evaluated on individual test images and display its predictions.
