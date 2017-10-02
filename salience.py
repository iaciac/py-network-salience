
# coding: utf-8


"""A Python module for extracting the salient links of a graph,
   as defined by Grady, D., Thiemann, C., and Brockmann, D. (2012).
"""


#    Copyright (C) 2016 by
#    Iacopo Iacopini <iacopini.iacopo@gmail.com>
#    All rights reserved.
#    BSD license.


import networkx as nx
from networkx.utils import accumulate
from networkx.utils import not_implemented_for
import numpy as np

__all__ = ['salience']

@not_implemented_for('directed')
@not_implemented_for('multigraph')
def _SPT(G, r, weight='weight'):
    """Returns the shortest-path tree (SPT) T(r) for a fixed reference node r of the graph G (see [1]).
        
        Args
        ----
        G: NetworkX graph
        
        r: node
            The reference node belonging to the graph G.
            The SPT summarizes the most effective routes from r to the rest of the network.
           
        weight: string (default='weight')
            Key for edge data used as the edge weight w_ij.
            The weight is used to compute the effective proximity d_ij = 1/_wij.
            WARNING: give the definition above, the weight value has to be a nonzero.
             
        Returns
        -------
        T: numpy.ndarray
            T(r) is a symmetric N × N matrix with elements tij ( r ) = 1 if the link ( i , j ) is part of
            at least one of the shortest paths and tij ( r ) = 0 if it is not. 
            
        References
        ----------    
        .. [1] Grady, Daniel, Christian Thiemann, and Dirk Brockmann.
           "Robust classification of salient links in complex networks."
           Nature communications 3 (2012): 864.

    """
    
    G = G.copy() 
    N = G.order()
    T = np.zeros((N,N))
    
    #Add the 'proximity' weight to each link (1/w) - WARNING: w has to be not zero!
    for i,j in G.edges():
        w = G[i][j][weight]
        G[i][j]['proximity'] = 1./w
    
    paths=nx.shortest_path(G,source=r, weight='proximity')
    #each path is a dictionary with key: r and value:the list of nodes in the path
    
    #Filling T based on the presence of a link in at least one of the shortest paths
    for k, path in paths.iteritems(): 
        for i in range(len(path)-1):
            T[path[i]][path[i+1]]=1 #updating the matrix T because there is a link
    return T

def salience(G,weight='weight'):
    """Returns the salience matrix of a graph G, defined as the average shortest-path tree (see [1]).
        
        Args
        ----
        G: NetworkX graph
           
        weight: string (default='weight')
            Key for edge data used as the edge weight w_ij.
            The weight is used to compute the effective proximity during the calculation
            of the shortest-path tree.
            
        Returns
        -------
        S: numpy.ndarray
            S is the salience of a network G, as a lienar superposition of all the shortest-path trees.
        
        References
        ----------    
        .. [1] Grady, Daniel, Christian Thiemann, and Dirk Brockmann.
           "Robust classification of salient links in complex networks."
           Nature communications 3 (2012): 864.

    """
    
    N = G.order()
    S = np.zeros((N,N))
    for n in G.nodes():
        S = S + _SPT(G,n,weight=weight)
    S = 1.*S/N
    return S    

