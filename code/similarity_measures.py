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


# Using the average covariates of the nodes' neighbours
def NBR(X, A, S):
    all_nodes = set(np.arange(A.shape[0]))
    no_cov_nodes = np.array(list(all_nodes - set(S)))
    X_hat = X.copy()
    neighbors = np.sum(A.A[no_cov_nodes][:, S], axis=1)
    
    X_hat[no_cov_nodes] = np.divide(np.sum(A.A[no_cov_nodes][:, S] * X[S], axis=1), neighbors, where=neighbors>0)
        
    return X_hat

# Personalised PageRank
def W_PPR(X, A, S, k):
    all_nodes = set(np.arange(A.shape[0]))
    no_cov_nodes = np.array(list(all_nodes - set(S)))
    X_hat = X.copy()
    
    gamma = np.exp(-.25)
    D = np.diag(A.dot(np.ones(A.shape[0])))
    D_inv = np.divide(1, D, where=D>0)
    M = (1 - gamma) * np.linalg.inv(np.eye(D.shape[0]) - gamma * A.dot(D_inv))
    W = (M + M.T) / 2
    
    neighbors = np.argsort(W[no_cov_nodes][:, S], axis=1)[:, -k:]
    weights = np.sum(np.take(W[no_cov_nodes][:, S], neighbors), axis=1)
    X_hat[no_cov_nodes] = np.divide(np.sum(np.take(W[no_cov_nodes][:, S]*X[S], neighbors), axis=1), weights, where=weights>0)
        
    return X_hat

# connection with the node degree
def Jaccard(X, A, S, k):
    all_nodes = set(np.arange(A.shape[0]))
    no_cov_nodes = np.array(list(all_nodes - set(S)))
    X_hat = X.copy()
    C = A.dot(A)
    d = A.dot(np.outer(np.ones(A.shape[0]), np.ones(A.shape[0])))
    W = np.array(C / (d + d.T - C))
    
    neighbors = np.argsort(W[no_cov_nodes][:, S], axis=1)[:, -k:]
    weights = np.sum(np.take(W[no_cov_nodes][:, S], neighbors), axis=1)
    X_hat[no_cov_nodes] = np.divide(np.sum(np.take(W[no_cov_nodes][:, S]*X[S], neighbors), axis=1), weights, where=weights>0)
       
    return X_hat

# Numver of common neighbours of the node for the W matrix
def CN(X, A, S, k):
    all_nodes = set(np.arange(A.shape[0]))
    no_cov_nodes = np.array(list(all_nodes - set(S)))
    X_hat = X.copy()
    
    W = A.dot(A).A
    
    neighbors = np.argsort(W[no_cov_nodes][:, S], axis=1)[:, -k:]
    weights = np.sum(np.take(W[no_cov_nodes][:, S], neighbors), axis=1)
    X_hat[no_cov_nodes] = np.divide(np.sum(np.take(W[no_cov_nodes][:, S]*X[S], neighbors), axis=1), weights, where=weights>0)
       
    return X_hat

