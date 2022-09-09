# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 12:01:31 2019

@author: an_fab
"""

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from matplotlib import pyplot as plt

from skimage import io

import math

import numpy as np

def sub2ind(array_shape, rows, cols):
    ind = rows*array_shape[1] + cols
    return ind

def ind2sub(array_shape, ind):
    rows = int(ind.astype('int') / array_shape[1])
    cols = ind % array_shape[1]
    return (rows, cols)

def show_img(img):
    width = 10.0
    height = img.shape[0]*width/img.shape[1]
    f = plt.figure(figsize=(width, height))
    plt.imshow(img)
    
def getImageGraph(img):
    
    rows = img.shape[0]
    cols = img.shape[1]
    
    dims = len(img.shape)
        
    c = 1 if dims == 2 else 3

    v = torch.tensor(np.reshape(img, [rows*cols,c]), dtype=torch.float)

    ind = np.asarray(range(0, rows*cols))
    pos = np.zeros((rows*cols,2))

    edges = []
    
    for i in ind:
        
        (y,x) = ind2sub((rows,cols), i)
        pos[i,0] = y
        pos[i,1] = x
        
            
        if x > 0 and x < cols - 1 and y > 0 and y < rows - 1:
                    
            edges.append([i, i + 1])
            edges.append([i, i - 1])
            edges.append([i, i - cols])
            edges.append([i, i + cols])

        if x == 0:
            
            if y == 0:
            
                edges.append([i, i+1])
                edges.append([i, i+cols])
            
            elif y == rows - 1:
            
                edges.append([i, i+1])
                edges.append([i, i-cols])                
                
            else:
            
                edges.append([i, i+1])
                edges.append([i, i-cols])   
                edges.append([i, i+cols])                  

        if x == cols - 1:
            
            if y == 0:
            
                edges.append([i, i-1])
                edges.append([i, i+cols])
            
            elif y == rows - 1:
            
                edges.append([i, i-1])
                edges.append([i, i-cols])                
                
            else:
            
                edges.append([i, i-1])
                edges.append([i, i-cols])   
                edges.append([i, i+cols])       

        if y == 0:
            
            if x == 0:
            
                edges.append([i, i+1])
                edges.append([i, i+cols])
            
            elif x == cols - 1:
            
                edges.append([i, i-1])
                edges.append([i, i+cols])                
                
            else:
            
                edges.append([i, i+1])
                edges.append([i, i-1])   
                edges.append([i, i+cols]) 
        
        if y == rows - 1:
            
            if x == 0:
            
                edges.append([i, i+1])
                edges.append([i, i-cols])
            
            elif x == cols - 1:
            
                edges.append([i, i-1])
                edges.append([i, i - cols])                
                
            else:
            
                edges.append([i, i+1])
                edges.append([i, i-1])   
                edges.append([i, i - cols]) 
     
    print('num edges: ' + str(len(edges)))
    #edges = np.sort(edges)
    unique_edges = [list(x) for x in set(tuple(x) for x in edges)]
    print('num unique edges: ' + str(len(unique_edges)))
    #print(unique_edges)
    
    edge_index = torch.tensor(unique_edges, dtype=torch.long).t().contiguous()
    
    edge_attr = np.zeros(len(unique_edges))
    
    beta = 0.1
    
    for i in range (0, edge_index.shape[1]):
        i1 = edge_index[0,i]
        i2 = edge_index[1,i]
        v1 = np.asarray(v[i1])
        v2 = np.asarray(v[i2])
        val = math.exp(-beta*np.sum(np.multiply(v1-v2,v1-v2)/c))
        edge_attr[i] = val 
    
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    pos = torch.tensor(pos, dtype=torch.int)

    data = Data(x = v, edge_index = edge_index, edge_attr = edge_attr, pos = pos)
    
    return data

    #edge_index = torch.tensor([np.asarray(e1), np.asarray(e2)], dtype=torch.long)
      
    #--> data.x: Node feature matrix with shape [num_nodes, num_node_features]
    #--> data.edge_index: Graph connectivity in COO format with shape [2, num_edges] and type torch.long
    #--> data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
    #data.y: Target to train against (may have arbitrary shape), e.g., node-level targets of shape [num_nodes, *] or graph-level targets of shape [1, *]
    #--> data.pos: Node position matrix with shape [num_nodes, num_dimensions]


path = 'test.jpg'

img = io.imread(path)
show_img(img)

graph = getImageGraph(img)