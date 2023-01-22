import numpy as np
import pandas as pd
import requests
import networkx as nx
from tqdm.notebook import tqdm
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import mean_squared_error
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs


def CN_VEC(X, A, S, k, return_time=False):
    """
    Args:
        X: array, matrix of all nodes' covariates, rows - objects, cols - covariates (features)
        A: array, Adjacency matrix of the graph (now weighted)
        S: array of indices, nodes, for which we know covariates
        k: number of neighbours
    Returns: X_hat, array, containing the estimation of the initial covariate matrix X, 
        supposedly, with the filled in missing values
    Comments:
        1. For now the algorithm is more suited to work with the csr_matrixes, which are
            know to usually represent the sparse data. Taking into account the restrictions
            on the network: the growth of edges is restricted between n**(1/3) and log(N) in O() terms, 
            is quite natural.
    """
    # mark the starting time for the future comparing
    start_time = time.time()
    
    # define the sets we are going to re-use later
    all_nodes = set(np.arange(A.shape[0]))
    empty_nodes = all_nodes - set(S)
    X_hat = X.copy()
    C = A.dot(A).toarray()
    #initially it used to be .A method, which seems to get old
    # I use the .toarray() method instead, as it does quite the same, but is much more readable
    # the whole sense of the algo lies in this calculation of the C-matrix: 
    # we observe the local attributes of the net.
    
    
    # iterate over the empty nodes, managing the logging
    for node_i in tqdm(empty_nodes):
        # first, we count up the C matrix for the i-th node
        Ci = np.ones_like(C) * C[node_i]
        Ci[:, node_i] = 0
        np.fill_diagonal(Ci, 0) # important to keep the diag free
        
        # now for the j_th, 0 <= i < j <= N
        Cj = C.copy()
        Cj[:, node_i] = 0
        np.fill_diagonal(Cj, 0) # important to keep the diag free
        
        # set the dists from the C-matrix as the paper states
        dists = np.sum((Ci**2 - 2) * (Ci >= 2) + (Cj**2 - 2) * (Cj >= 2) - 2 * Ci * Cj, axis=1)[S]
        
        idxs = np.argsort(dists)[:k] # that is a costly operation, work on it if possible
        # however, comparing with the O(N**3) operation with matrixes, it is insignificant
        ktop = np.array(S)[idxs] # seems like we do not need the sorting here, we may just find the top-K elements
        X_hat[node_i] = np.mean(X[ktop])
        
    # a bit of logging, just to fulfill the curiousity and for the future comparance
    print(f'The time, taken to build the X_hat matrix is {time.time() - start_time}')
    if return_time:
        return X_hat, time.time() - start_time
    return X_hat


# Algorithm 2
def SVD_RBF(X, A, S, theta, d):
    """
    Args:
        X: array, Covariates of all nodes. For nodes which covariates are unknown array contains NaN
        A: array, Adjacency matrix of the graph (now weighted)
        S: array of indexes, Set of nodes with known covariates
        theta: bandwidth (restriction on the eigs computed <= N - 1, where A in N x N)
        d: rank of matrix
    Return:
        X_hat: array, the estimation of the initial matrix X with filled-in gaps via SVD_RBF
    
    """

    E, U = eigs(A, d)
    V = np.real(U) * np.abs(np.real(E))**(1/2) #returns the product of the real parts of the corresponding eig-matrixes
    # now work with the nodes, enumerate them casually
    all_nodes = set(np.arange(A.shape[0]))
    # select the nodes with no covariates provided
    no_cov_nodes = np.array(list(all_nodes - set(S)))
    # initialize with the copy of X
    X_hat = X.copy()
    # compute the distances via the RBF kernel, following the algorithm
    dists = rbf_kernel(V, V, gamma=1/2/(theta**2))
    # fill the missing covariates with the normalized vals, following the algorithm
    X_hat[no_cov_nodes] = np.sum(dists[no_cov_nodes][:, S] * X[S], axis=1) / np.sum(dists[no_cov_nodes][:, S], axis=1)
        
    return X_hat