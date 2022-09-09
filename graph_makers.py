# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 14:41:30 2020

@author: an_fab
"""

import torch
import numpy as np

from helpers import ind2sub
from torch_geometric.data import Data

from skimage import color
from skimage.segmentation import slic
from skimage.measure import regionprops
from skimage.future import graph

def GraphFromImage(img, beta):
    
    Y, X = img.shape[1], img.shape[0]
    
    c = 3 if len(img.shape)==3 else 1
    
    v = np.reshape(img, [X*Y, c])
    v = v/255
    
    ind = list(range(0, X*Y))
    
    edges = []
    pos = []
    
    for i in ind:
        
        y, x = ind2sub([Y, X], i)
        pos.append([y, x])
        
        if x > 0 and x < X-1 and y > 0 and y < Y-1:
           
           edges.append([i, i - 1])
           edges.append([i, i + 1])
           edges.append([i, i - X])
           edges.append([i, i + X])
        
        if x == 0:
            
            edges.append([i, i + 1])
            
            if y == 0:
                        
                edges.append([i, i + X])
            
            elif y == Y-1:
                
                edges.append([i, i - X])
            
            else:
                       
               edges.append([i, i - X])
               edges.append([i, i + X])
        
    
        if x == X-1:
            
            edges.append([i, i - 1])
            
            if y == 0:
            
                edges.append([i, i + X])
                    
            elif y == Y-1:
            
                edges.append([i, i - X])
                
            else: 
            
                edges.append([i, i - X])
                edges.append([i, i + X])
                
        if y == 0:
            
            edges.append([i, i + X])
            
            if x == 0:
                
                edges.append([i, i + 1])
                    
            elif x == X-1:
                
                edges.append([i, i - 1])
            
            else:
                
                edges.append([i, i - 1])
                edges.append([i, i + 1])

        if y == Y-1:
            
            edges.append([i, i - X])
            
            if x == 0:
                
                edges.append([i, i + 1])
                    
            elif x == X-1:
                
                edges.append([i, i - 1])
            
            else:
                
                edges.append([i, i - 1])
                edges.append([i, i + 1])
                
    print('num edges: ' + str(len(edges)))
    unique_edges = [list(x) for x in set(tuple(x) for x in edges)]
    print('num unique edges: ' + str(len(unique_edges)))

    #beta = 0.01
    # alpha = 0.6
    # beta = 0.4
    # theta_a = 0.50
    # theta_b = 0.10
    # theta_g = 2
    
    edge_attr = []
        
    for e in unique_edges:
        
        v1 = np.asarray(v[e[0]])
        v2 = np.asarray(v[e[1]])
        p1 = np.asarray(pos[e[0]])
        p2 = np.asarray(pos[e[1]])
        
        #dist1 = np.exp(-np.sum(np.multiply(p1-p2,p1-p2))/(2*theta_a**2)-np.sum(np.multiply(v1-v2,v1-v2))/(2*theta_b**2))
        #dist2 = np.exp(-np.sum(np.multiply(p1-p2,p1-p2))/(2*theta_g**2))
        #dist = alpha*dist1 + beta*dist2
        dist = np.exp(-beta*np.sum(np.multiply(v1-v2,v1-v2)))
        edge_attr.append(dist)
        
        #edge_attr.append(1)

    #v = np.concatenate((v,np.asarray(pos)/max(X,Y)), axis=1)
    edge_index = torch.tensor(unique_edges, dtype=torch.long).t().contiguous()
    v = torch.from_numpy(v).float()
    pos = torch.tensor(pos, dtype = torch.long)    
    edge_attr = torch.tensor(edge_attr, dtype = torch.float)
    graph = Data(x=v, edge_index = edge_index, pos = pos, edge_attr = edge_attr)
    
    return graph

def GraphFromImage2(img, alpha, beta, theta_a, theta_b, theta_g):
    
    Y, X = img.shape[1], img.shape[0]
    
    c = 3 if len(img.shape)==3 else 1
    
    v = np.reshape(img, [X*Y, c])
    v = v/255
    
    ind = list(range(0, X*Y))
    
    edges = []
    pos = []
    
    for i in ind:
        
        y, x = ind2sub([Y, X], i)
        pos.append([y, x])
        
        if x > 0 and x < X-1 and y > 0 and y < Y-1:
           
           edges.append([i, i - 1])
           edges.append([i - 1, i]) #
           
           edges.append([i, i + 1])
           edges.append([i + 1, i]) #
           
           edges.append([i, i - X])
           edges.append([i - X, i]) #
           
           edges.append([i, i + X])
           edges.append([i + X, i]) #
           
        
        if x == 0:
            
            edges.append([i, i + 1])
            edges.append([i + 1, i]) #
            
            if y == 0:
                        
                edges.append([i, i + X])
                edges.append([i + X, i]) #
                
            
            elif y == Y-1:
                
                edges.append([i, i - X])
                edges.append([i - X, i]) #
            
            else:
                       
               edges.append([i, i - X])
               edges.append([i - X, i]) #
               
               edges.append([i, i + X])
               edges.append([i + X, i]) #
        
    
        if x == X-1:
            
            edges.append([i, i - 1])
            edges.append([i - 1, i]) #
            
            if y == 0:
            
                edges.append([i, i + X])
                edges.append([i + X, i]) #

                    
            elif y == Y-1:
            
                edges.append([i, i - X])
                edges.append([i - X, i]) #
                
            else: 
            
                edges.append([i, i - X])
                edges.append([i - X, i]) # 
                
                edges.append([i, i + X])
                edges.append([i + X, i]) #
                
        if y == 0:
            
            edges.append([i, i + X])
            edges.append([i + X, i]) #
            
            if x == 0:
                
                edges.append([i, i + 1])
                edges.append([i + 1, i]) #
                    
            elif x == X-1:
                
                edges.append([i, i - 1])
                edges.append([i - 1, i]) #
                
            else:
                
                edges.append([i, i - 1])
                edges.append([i - 1, i]) #
                
                edges.append([i, i + 1])
                edges.append([i + 1, i]) #

        if y == Y-1:
            
            edges.append([i, i - X])
            edges.append([i - X, i]) #
            
            if x == 0:
                
                edges.append([i, i + 1])
                edges.append([i + 1, i]) #
                    
            elif x == X-1:
                
                edges.append([i, i - 1])
                edges.append([i - 1, i]) #
            
            else:
                
                edges.append([i, i - 1])
                edges.append([i - 1, i]) #
                
                edges.append([i, i + 1])
                edges.append([i + 1, i]) 
                
    print('num edges: ' + str(len(edges)))
    unique_edges = [list(x) for x in set(tuple(x) for x in edges)]
    print('num unique edges: ' + str(len(unique_edges)))

    #beta = 0.01
    # alpha = 0.6
    # beta = 0.4
    # theta_a = 0.50
    # theta_b = 0.10
    # theta_g = 2
    
    edge_attr = []
        
    for e in unique_edges:
        v1 = np.asarray(v[e[0]])
        v2 = np.asarray(v[e[1]])
        p1 = np.asarray(pos[e[0]])
        p2 = np.asarray(pos[e[1]])
        
        #dist1 = np.exp(-np.sum(np.multiply(p1-p2,p1-p2))/(2*theta_a**2)-np.sum(np.multiply(v1-v2,v1-v2))/(2*theta_b**2))
        #dist2 = np.exp(-np.sum(np.multiply(p1-p2,p1-p2))/(2*theta_g**2))
        #dist = alpha*dist1 + beta*dist2
        dist = np.exp(-beta*np.sum(np.multiply(v1-v2,v1-v2)))
        dist2 = np.sqrt(np.sum(np.multiply(p1-p2,p1-p2)))
        #dist = dist  / np.exp(-np.sum(np.multiply(p1-p2,p1-p2)))
        edge_attr.append(dist/dist2)
        
        #edge_attr.append(1)

    #v = np.concatenate((v,np.asarray(pos)/max(X,Y)), axis=1)
    edge_index = torch.tensor(unique_edges, dtype=torch.long).t().contiguous()
    v = torch.from_numpy(v).float()
    pos = torch.tensor(pos, dtype = torch.long)    
    edge_attr = torch.tensor(edge_attr, dtype = torch.float)
    graph = Data(x=v, edge_index = edge_index, pos = pos, edge_attr = edge_attr)
    
    return graph

def GraphFromSuperpixels(img):
    
    Y, X = img.shape[1], img.shape[0]
    N = int(0.1*Y*X)

    
    sp = slic(img, n_segments=N, sigma = 3, convert2lab=True)
    sp = sp + 1
    
    regions = regionprops(sp)
    
    RAG = graph.rag_mean_color(img, sp)
    
    for region in regions:
        RAG.nodes[region['label']]['centroid'] = region['centroid']
    
    edges = []
    pos = []
    v = []
    
    for i in range (1, np.max(sp)+1):
        
        c = RAG.nodes[i]['mean color']
        c = c/255
        v.append(c)
        
        x = RAG.nodes[i]['centroid'][1]
        y = RAG.nodes[i]['centroid'][0]
        
        pos.append([y, x])
        
    for e in RAG.edges:
        edges.append([e[0]-1, e[1]-1])
    
    print('num unique edges: ' + str(len(edges)))
    
    #beta = 0.01
    beta = 100
    edge_attr = []
    
    for e in edges:
        v1 = np.asarray(v[e[0]])
        v2 = np.asarray(v[e[1]])
        #dist = np.exp(-beta*np.sum(np.multiply(v1-v2,v1-v2)))
        #edge_attr.append(dist)
        edge_attr.append(1)

    #v = np.concatenate((v,np.asarray(pos)/max(X,Y)), axis=1)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    v = torch.from_numpy(np.asarray(v)).float()
    pos = torch.tensor(pos, dtype = torch.long)    
    edge_attr = torch.tensor(edge_attr, dtype = torch.float)
    _graph = Data(x=v, edge_index = edge_index, pos = pos, edge_attr = edge_attr)
    
    return _graph, sp