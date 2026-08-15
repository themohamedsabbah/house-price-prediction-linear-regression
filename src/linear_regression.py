import numpy as np

def compute_gradient(X, y, w, b):
    """
    Computes the gradient for linear regression.

    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters
      b (scalar)       : model parameter
    Returns:
      dj_db (scalar)      : gradient of the cost w.r.t. b
      dj_dw (ndarray (n,)): gradient of the cost w.r.t. w
    """
    m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):
        prediction = np.dot(X[i], w) + b
        error = prediction - y[i]
        for j in range(n):
            dj_dw[j] += error * X[i, j]
        dj_db += error

    dj_dw /= m
    dj_db /= m
    return dj_db, dj_dw