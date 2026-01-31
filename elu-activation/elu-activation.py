import numpy as np

def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    res = []
    for i in x:
        # x > 0
        if i > 0:
            res.append(i)

        # x <= 0
        else:
            res.append(alpha * (np.exp(i) - 1))

    return res