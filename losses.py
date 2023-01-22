import numpy as np

def binary_cross_entropy(y_true, y_pred):
    return np.mean(-y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_deriv(y_true, y_pred):
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)

def mean_squared_error(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mean_squared_error_deriv(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)
