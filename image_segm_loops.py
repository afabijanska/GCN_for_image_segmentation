# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 10:42:34 2019

@author: an_fab
"""
 
import os
import glob

import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphUNet
from torch_geometric.utils import dropout_adj
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, ChebConv, SplineConv, GATConv, SAGEConv, AGNNConv, ARMAConv, GraphUNet, GatedGraphConv  # noqa
from torch_geometric.nn import DNAConv

import random
import numpy as np
import matplotlib.pyplot as plt

from skimage import io, filters

from torch_geometric.data import Data

from helpers import show_img
from graph_makers import GraphFromImage
   
#----------- some settings -------------

bkgLabel = int("{:08b}".format(0)+"{:08b}".format(0)+"{:08b}".format(0), 2)
mode = 'rect' #or 'scribb'

#----------- net model -------------

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.conv1 = GCNConv(3, 16)
        #self.conv3 = GCNConv(16,16)
        #self.conv2 = GCNConv(16, 3)
        self.conv1 = ChebConv(data.num_features, 16, K=2)
        self.conv2 = ChebConv(16, data.num_classes, K=2)

        # self.conv1 = SAGEConv(data.num_features, 16, normalize = True)
        # self.conv3 = SAGEConv(16, 16, normalize = True)
        # self.conv2 = SAGEConv(16, 2, normalize = True)
        
        # self.conv1 = GatedGraphConv(data.num_features, 32)
        # self.conv2 = GatedGraphConv(32, data.num_features)

        #self.conv1 = SplineConv(data.num_features, 32, dim=2, kernel_size=5, aggr='max')
        #self.conv3 = SplineConv(32, 32, dim=2, kernel_size = 3, aggr = 'max')
        #self.conv2 = SplineConv(32, 2, dim=2, kernel_size=5, aggr='max')
        
        #self.reg_params = self.conv1.parameters()
        #self.non_reg_params = self.conv2.parameters()

    #forward(x, edge_index, size=None)
    def forward(self):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        # x = F.dropout(x, training=self.training)
        # x = F.relu(self.conv3(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)

        return F.log_softmax(x, dim=1)

# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.lin1 = torch.nn.Linear(data.num_features, 16)
#         self.prop1 = AGNNConv(requires_grad=False)
#         self.prop2 = AGNNConv(requires_grad=True)
#         self.lin2 = torch.nn.Linear(16, 2)

#     def forward(self):
#         x = F.dropout(data.x, training=self.training)
#         x = F.relu(self.lin1(x))
#         x = self.prop1(x, data.edge_index)
#         x = self.prop2(x, data.edge_index)
#         x = F.dropout(x, training=self.training)
#         x = self.lin2(x)
#         return F.log_softmax(x, dim=1)
    
# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = GATConv(data.num_features, 8, heads=8, dropout=0.6)
#         # On the Pubmed dataset, use heads=8 in conv2.
#         self.conv2 = GATConv(8 * 8, 2, heads=1, concat=True, dropout=0.6)

#     def forward(self):
#         x = F.dropout(data.x, p=0.6, training=self.training)
#         x = F.elu(self.conv1(x, data.edge_index))
#         x = F.dropout(x, p=0.6, training=self.training)
#         x = self.conv2(x, data.edge_index)
#         return F.log_softmax(x, dim=1)
    

# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()

#         self.conv1 = ARMAConv(
#             data.num_features,
#             16,
#             num_stacks=3,
#             num_layers=2,
#             shared_weights=True,
#             dropout=0.25)

#         self.conv2 = ARMAConv(
#             16,
#             2,
#             num_stacks=3,
#             num_layers=2,
#             shared_weights=True,
#             dropout=0.25,
#             act=None)

#     def forward(self):
#         x, edge_index = data.x, data.edge_index
#         x = F.dropout(x, training=self.training)
#         x = F.relu(self.conv1(x, edge_index))
#         x = F.dropout(x, training=self.training)
#         x = self.conv2(x, edge_index)
#         return F.log_softmax(x, dim=1)

# class Net(torch.nn.Module):
#     def __init__(self,
#                  in_channels,
#                  hidden_channels,
#                  out_channels,
#                  num_layers,
#                  heads=1,
#                  groups=1):
#         super(Net, self).__init__()
#         self.hidden_channels = hidden_channels
#         self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
#         self.convs = torch.nn.ModuleList()
#         for i in range(num_layers):
#             self.convs.append(
#                 DNAConv(
#                     hidden_channels, heads, groups, dropout=0.8, cached=True))
#         self.lin2 = torch.nn.Linear(hidden_channels, out_channels)

#     def reset_parameters(self):
#         self.lin1.reset_parameters()
#         for conv in self.convs:
#             conv.reset_parameters()
#         self.lin2.reset_parameters()

#     def forward(self, x, edge_index):
#         x = F.relu(self.lin1(x))
#         x = F.dropout(x, p=0.5, training=self.training)
#         x_all = x.view(-1, 1, self.hidden_channels)
#         for conv in self.convs:
#             x = F.relu(conv(x_all, edge_index))
#             x = x.view(-1, 1, self.hidden_channels)
#             x_all = torch.cat([x_all, x], dim=1)
#         x = x_all[:, -1]
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.lin2(x)
#         return torch.log_softmax(x, dim=1)

# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         pool_ratios = [0.5]
#         self.unet = GraphUNet(data.num_features, 16, 2,
#                               depth=2, pool_ratios=pool_ratios, sum_res = True)

#     def forward(self):
#         edge_index, _ = dropout_adj(
#             data.edge_index, p=0.2, force_undirected=True,
#             num_nodes=data.num_nodes, training=self.training)
#         x = F.dropout(data.x, p=0.92, training=self.training)

#         x = self.unet(x, edge_index)
#         return F.log_softmax(x, dim=1)


#----------- data paths ------------

main_dir = 'C:\\Users\\an_fab\\Desktop\\graph_segm\\berkeley'

#dir_gt = 'C:\\Users\\an_fab\\Desktop\\graph_segm\\microsoft'+'\\gt\\'
dir_img = 'C:\\Users\\an_fab\\Desktop\\graph_segm\\berkeley'+'\\org\\'
dir_results = 'C:\\Users\\an_fab\\Desktop\\graph_segm\\berkeley'+'\\results\\'
dir_scrib = 'C:\\Users\\an_fab\\Desktop\\graph_segm\\berkeley'+'\\scribbles\\'
#dir_rect = 'C:\\Users\\an_fab\\Desktop\\graph_segm\\microsoft'+'\\rects\\'

for name in glob.glob(dir_img+'/*'):
    
    path_img = name
    
    head, tail = os.path.split(name)
    filename, file_extension = os.path.splitext(tail)
    path_labels = dir_scrib + filename +'.bmp'
    #path_gt = dir_gt + filename +'.bmp'
    path_result = dir_results + filename +'.bmp'
    #path_rect = dir_rect + filename + '.bmp'

    #----------- read inputs ------------

    img = io.imread(path_img)
    labels = io.imread(path_labels)
    #gt = io.imread(path_gt)

    Y, X = img.shape[1], img.shape[0]
    c = 3 if len(img.shape)==3 else 1
    
    # if mode == 'scrib':
    #     rect = io.imread(path_rect)
    # else:
    #     rect = np.zeros((X,Y,3), dtype = np.uint8)
        
    rect = np.zeros((X,Y,3), dtype = np.uint8)
    show_img(img)
    show_img(labels)

    img = filters.median(img, behavior='ndimage')
    
    labels = np.maximum(labels, rect)

    #----------- get image graph ------------    


    theta_a = 0.50
    theta_b = 0.10
    theta_g = 2
    
    
    for alpha in np.arange(0,1.1,0.25):
        
        for theta_a in np.arange(1, 21, 5):
            
            for theta_b in np.arange(1, 21, 5):
                
                for theta_g in np.arange(1, 21, 5):
        
                    beta = 1-alpha
                    
                    path_result = dir_results + filename + '_alpha_'+str('%.1f'%alpha)+'_beta_'+str('%.1f'%beta)+'_a_'+str('%.1f'%theta_a)+'_b_'+str('%.1f'%theta_b)+'_g_'+str('%.1f'%theta_g)+'.bmp'
                    
                    data = GraphFromImage(img, alpha, beta, theta_a, theta_b, theta_g)
                    print(data)
                
                    #----------- get labels ------------ 
                
                    l = np.reshape(labels, (Y*X, 3))
                    L = np.zeros((Y*X), dtype = int)
                
                    for i in range(0,len(l)):
                        a = "{:08b}".format(l[i,0])+"{:08b}".format(l[i,1])+"{:08b}".format(l[i,2])
                        L[i] = int(a, 2)
                
                    l = np.unique(L)
                    num_classes = len(l)-1
                    print('num labels: ' + str(len(l)-1))
                
                    #----------- prepare train/test set ------------ 
                
                    test_mask = np.zeros((X*Y), dtype = 'bool')
                    train_mask = np.zeros((X*Y), dtype = 'bool')
                    val_mask = np.zeros((X*Y), dtype = 'bool')
                    y = np.zeros((X*Y), dtype = int)
                
                    val = -1
                    
                    for lab in l:
                        
                        indx = np.where(lab == L)
                        
                        if lab == bkgLabel:
                        
                            y[indx]  =  random.randint(0,num_classes-1)
                            train_mask[indx] = False
                            test_mask[indx] = True
                            #val_mask[indx] = False
                        
                        else:
                                    
                            val = val + 1
                            y[indx] = val
                            train_mask[indx] = True
                            test_mask[indx] = False
                            #val_mask[indx] = False
                                
                    test_mask = torch.from_numpy(np.asarray(test_mask)).bool()
                    train_mask = torch.from_numpy(np.asarray(train_mask)).bool()
                    val_mask = torch.from_numpy(np.asarray(val_mask)).bool()
                    y = torch.from_numpy(np.asarray(y)).long()     
                            
                    data.test_mask = test_mask
                    data.train_mask = train_mask
                    data.val_mask = val_mask
                    data.num_classes = num_classes
                    data.y = y
                
                    print(data)
                
                    #----------- train classfier ------------ 
                    
                    #torch.cuda.empty_cache()
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                     
                    #device = 'cpu'       
                    
                    #model, data = Net().to(device), data.to(device)
                    
                    model = Net()
                    
                    model, data = model.to(device), data.to(device)
                   
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)
                    #optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)
                    
                    def train():
                        model.train()
                        optimizer.zero_grad()
                        F.nll_loss(model()[data.train_mask], data.y[data.train_mask]).backward()
                        optimizer.step()
                    
                    def test():
                        model.eval()
                        logits, accs = model(), []
                        for _, mask in data('train_mask'):
                            pred = logits[mask].max(1)[1]
                            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
                            accs.append(acc)
                        return accs
                    
                    best_val_acc = 0
                    patience = 500
                    counter = 0
                        
                    for epoch in range(1, 5001):
                        train()
                        train_acc = test()
                        
                        a = float(train_acc[0])
                        
                        if a > best_val_acc:
                             torch.save(model, 'entire_model.pth')
                             best_val_acc = a
                             counter = 0
                        else:
                             counter = counter + 1
                        
                        if counter >= patience:
                            break
                        
                        if epoch % 100 == 0:
                            print('File:' + tail + ', classes: ' +  str(len(l)-1)  + ', epoch: ' + str(epoch) +': acc: ' + str(train_acc))
                        
                        
                    model = torch.load('entire_model.pth')
                    logits, accs = model(), []
                    
                    for _, mask in data('test_mask'):
                        pred = logits[mask].max(1)[1]
                    
                    result = np.zeros(Y*X)
                    result[data.test_mask.cpu()] = pred.cpu()
                    result = np.reshape(result, [X, Y])
                    #plt.imshow(result)
                    
                    result = np.zeros(Y*X)
                    result[data.test_mask.cpu()] = pred.cpu()
                    result[data.train_mask.cpu()] = data.y.cpu()[data.train_mask.cpu()]
                    result = np.reshape(result, [X,Y])
                    #plt.imshow(result)
                    
                    #save result
                    result = 255*result/(len(l)-1)
                    io.imsave(path_result, result.astype(int))
        
    #result = np.zeros(321*481)
    #result[data.test_mask] = pred
    #result[data.train_mask] = data.y[data.train_mask]
    #result = np.reshape(result, [321, 481])
    #plt.imshow(result)
    
    #plt.imshow(labels)
    #plt.imshow(img)