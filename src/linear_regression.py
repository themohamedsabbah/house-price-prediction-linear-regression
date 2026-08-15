import numpy as np
import copy
import math
from cost_functions import compute_cost

COST = "cost"
PARAM = "params"
GRADES = "grads"
ITERTATIONS = "iter"

def run_gradient_descent(X,y,iterations=1000, alpha = 1e-6):
    _,n = X.shape
    # initialize parameters
    initial_w = np.zeros(n)
    initial_b = 0
    w_out, b_out, hist_out = gradient_descent_houses(X, y, initial_w, initial_b, compute_cost, compute_gradient, alpha, iterations)
    print(f"w,b found by gradient descent: w: {w_out}, b: {b_out:0.2f}")
    return w_out, b_out, hist_out
     
def gradient_descent_houses(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    hist={}
    hist[COST] = []; hist[PARAM] = []; hist[GRADES]=[]; hist[ITERTATIONS]=[];

    w = copy.deepcopy(w_in)
    b = copy.deepcopy(b_in)

    save_interval = np.ceil(num_iters/10000) # prevent resource exhaustion for long runs

    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i == 0 or i % save_interval == 0:
                    hist[COST].append(cost_function(X, y, w, b))
                    hist[PARAM].append([w,b])
                    hist[GRADES].append([dj_dw,dj_db])
                    hist[ITERTATIONS].append(i)
        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            cst = cost_function(X, y, w, b)
            print(f"{i:9d} {cst:0.5e} {w[0]: 0.1e} {w[1]: 0.1e} {w[2]: 0.1e} {w[3]: 0.1e} {b: 0.1e} {dj_dw[0]: 0.1e} {dj_dw[1]: 0.1e} {dj_dw[2]: 0.1e} {dj_dw[3]: 0.1e} {dj_db: 0.1e}")
        
    return w, b, hist #return w,b and history for graphin
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

def zscore_normalize_features(X,rtn_ms=False):
    """
    returns z-score normalized X by column
    Args:
      X (ndarray (m,n)) :
    Returns
      X_norm (ndarray (m,n)): input normalized by column
    """
    mu     = np.mean(X,axis=0)
    sigma  = np.std(X,axis=0)
    X_norm = (X - mu)/sigma

    if rtn_ms:
        return X_norm, mu, sigma
    return X_norm