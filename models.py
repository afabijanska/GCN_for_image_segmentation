# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 17:34:42 2020

@author: an_fab
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphUNet
from torch_geometric.utils import dropout_adj
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv, SplineConv  # noqa


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.conv1 = GCNConv(3, 16)
        #self.conv3 = GCNConv(16,16)
        #self.conv2 = GCNConv(16, 3)
        self.conv1 = ChebConv(data.num_features, 16, K=3)
        self.conv2 = ChebConv(16, data.num_features, K=3)

        #self.conv1 = SplineConv(data.num_features, 32, dim=2, kernel_size=3, aggr='mean')
        #self.conv2 = SplineConv(32, data.num_features, dim=2, kernel_size=3, aggr='mean')
        
        self.reg_params = self.conv1.parameters()
        self.non_reg_params = self.conv2.parameters()

    def forward(self):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        #x = F.relu(self.conv3(x, edge_index, edge_weight))
        #x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)