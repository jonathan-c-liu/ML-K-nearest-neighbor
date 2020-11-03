import numpy as np 

def euclidean_distances(X, Y):
    """Compute pairwise Euclidean distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Euclidean distance between two rows.
    
    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Euclidean distances between rows of X and rows of Y.
    """
    D = np.zeros((X.shape[0], Y.shape[0]))
    for M in range(X.shape[0]):
        for N in range(Y.shape[0]):
            D[M, N] = np.sqrt(np.sum(np.square(np.subtract(X[M, :], Y[N, :]))))
    return D


def manhattan_distances(X, Y):
    """Compute pairwise Manhattan distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Manhattan distance between two rows.
    
    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Manhattan distances between rows of X and rows of Y.
    """
    D = np.zeros((X.shape[0], Y.shape[0]))
    for M in range(X.shape[0]):
        for N in range(Y.shape[0]):
            D[M, N] = np.sum(np.abs(np.subtract(X[M, :], Y[N, :])))
    return D


def cosine_distances(X, Y):
    """Compute Cosine distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Cosine distance between two rows.
    
    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Cosine distances between rows of X and rows of Y.
    """
    D = np.zeros((X.shape[0], Y.shape[0]))
    for M in range(X.shape[0]):
        for N in range(Y.shape[0]):
            D[M, N] = 1 - np.dot(X[M, :], Y[N, :]) / (np.linalg.norm(X[M, :]) * np.linalg.norm(Y[N, :]))
    return D