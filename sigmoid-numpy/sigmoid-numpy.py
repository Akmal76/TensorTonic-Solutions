import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    sigmoid = np.vectorize(lambda x: 1 / (1 + np.exp(-x)))
    return sigmoid(np.array(x))