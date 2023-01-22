import os
import numpy as np
import pandas as pd
import networkx as nx
from scipy.sparse import csr_matrix, diags

class CoreDataset:
    def __init__(self,
                 data_dir = 'data/processed/cora',
                 project_dir = '/Users/u_m1anq/Desktop/Thesis-code/'):
        self.full_path = os.path.join(project_dir, data_dir)
        self.X_nodes, self.X_edges, self.mapper, self.reverse_mapper = self._read_data()
        self.adjacency = self._get_adjacency()
        
    def _read_data(self, return_freqs=False):
        
        # the nodes part    
        node_column_names =  ["w_{}".format(ii) for ii in range(1433)] + ["subject"]
        cora_nodes = pd.read_csv(
            os.path.join(self.full_path, "cora.content"),
            sep='\t',
            header=None,
            names=node_column_names)
        # depending on whether we need additional covariates
        # those are certain words' frequences
        if not return_freqs:
            X_nodes = cora_nodes['subject'].reset_index()
        else:
            X_nodes = cora_nodes
            
        # the edges part
        cora_edges = pd.read_csv(
            os.path.join(self.full_path, "cora.cites"),
            sep='\t',
            header=None,
            names=["target", "source"]
            )       
        rows, cols = cora_edges['target'], cora_edges['source']
        
        # the info is somewhat encoded, to be clear we are to map the names
        keys = list(set(rows)|set(cols))
        keys.sort()
        
        mapper = {}
        for i, k in enumerate(keys):
            mapper[k] = i

        # we also are to create the reverse-mapper in case we need the backward-turn
        reverse_mapper = {}
        for k, v in mapper.items():
            reverse_mapper[v] = k
        
        
            
        # now use the forward mapper to keep the new nodenames
        rows = cora_edges['target'].map(mapper)
        cols = cora_edges['source'].map(mapper)
        X_edges = list(zip(rows, cols))
        
        # also use the same mapper on the X_nodes_info
        X_nodes['index'] = X_nodes['index'].map(mapper)
        X_nodes.rename({'index': 'node'}, axis=1, inplace=True)
        
        return X_nodes, X_edges, mapper, reverse_mapper
            
    
    def _get_adjacency(self, csr=True):
        """
        Effectively preprocesses the explicit edgelist into scipy.sparse.csr_matric (m.b. coo faster?)
        """
        size = len(set([n for e in self.X_edges for n in e])) 
        # make an empty adjacency list  
        adjacency = [[0]*size for _ in range(size)]
        # populate the list for each edge
        for sink, source in self.X_edges:
            adjacency[sink][source] = 1

        # turn the datatype
        return csr_matrix(adjacency) if csr else np.array(adjacency)
    