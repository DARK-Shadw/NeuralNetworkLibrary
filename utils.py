import numpy as np
import pandas as pd

def permute_data(X, Y):
    # Create a random permutation of indices
    perm = np.random.permutation(len(X))

    # Check if X and Y are pandas DataFrame, otherwise use them as NumPy arrays
    if isinstance(X, pd.DataFrame):
        return X.iloc[perm].values, Y.iloc[perm].values
    else:
        return X[perm], Y[perm]

def generate_batch(X, Y, size):
    assert X.shape[0] == Y.shape[0]

    N = X.shape[0]

    for ii in range(0, N, size):
        X_batch, Y_batch = X[ii:ii+size], Y[ii:ii+size]
        yield X_batch, Y_batch