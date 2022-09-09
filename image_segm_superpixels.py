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
import matplotlib
import matplotlib.pyplot as plt
from skimage.segmentation import slic

from skimage import io, filters, color

from torch_geometric.data import Data

from helpers import show_img
from graph_makers import GraphFromImage, GraphFromSuperpixels
from skimage.future import graph
from skimage.filters import median
from skimage.color import label2rgb
   
#----------- some settings -------------

bkgLabel = int("{:08b}".format(0)+"{:08b}".format(0)+"{:08b}".format(0), 2)
mode = 'rect' #or 'scribb'

#----------- net model -------------

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(data.num_features, 16)
        self.conv3 = GCNConv(16,16)
        self.conv2 = GCNConv(16, data.num_classes)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_classes, K=2)

        #self.conv1 = SAGEConv(data.num_features, 16, normalize = True)
        #self.conv3 = SAGEConv(16, 16, normalize = True)
        #self.conv2 = SAGEConv(16, 2, normalize = True)
        
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
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv3(x, edge_index, edge_weight))
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

main_dir = 'C:\\Users\\an_fab\\Desktop\\graph_segm\\microsoft'

# dir_gt = 'C:\\Users\\an_fab\\Desktop\\graph_segm\\microsoft'+'\\gt\\'
# dir_img = 'C:\\Users\\an_fab\\Desktop\\graph_segm\\microsoft'+'\\org\\'
# dir_results = 'C:\\Users\\an_fab\\Desktop\\graph_segm\\microsoft'+'\\results\\'
# dir_scrib = 'C:\\Users\\an_fab\\Desktop\\graph_segm\\microsoft'+'\\scribbles\\'
# dir_rect = 'C:\\Users\\an_fab\\Desktop\\graph_segm\\microsoft'+'\\rects\\'


dir_img = 'C:\\Users\\an_fab\\Desktop\\VOCdevkit\\VOC2012\\JPEGImages\\'
dir_results = 'C:\\Users\\an_fab\\Desktop\\VOCdevkit\\VOC2012\\SegmentationGraphs\\'
dir_scrib = 'C:\\Users\\an_fab\\Desktop\\VOCdevkit\\VOC2012\\AutoGeneratedScribbles\\'
dir_gt = 'C:\\Users\\an_fab\\Desktop\\VOCdevkit\\VOC2012\\SegmentationObject\\'

for name in glob.glob(dir_img+'/*'):
    
    path_img = name
    
    head, tail = os.path.split(name)
    filename, file_extension = os.path.splitext(tail)
    path_labels = dir_scrib + filename +'.bmp'
    path_gt = dir_gt + filename +'.png'
    path_result = dir_results + filename +'.bmp'
    #path_rect = dir_rect + filename + '.bmp'
   
    #----------- read inputs ------------

    img = io.imread(path_img)
    #labels = io.imread(path_labels)

    
    
    try:
        
        labels = io.imread(path_labels)
        gt = io.imread(path_gt)
        
    except OSError:
        print('skipping')
        continue

    
        
    print(path_img)
        
    Y, X = img.shape[1], img.shape[0]
    c = 3 if len(img.shape)==3 else 1
    
    # mode == 'scrib'
     
    # if mode == 'scrib':
    #     rect = io.imread(path_rect)
    # else:
    #     rect = np.zeros((X,Y,3), dtype = np.uint8)
        
    show_img(img)
    show_img(labels)
    
    #img = median(img, behavior='ndimage')

    #----------- get image graph ------------  
    
    data, sp = GraphFromSuperpixels(img)
    print(data)
    N = np.max(sp)
    
    label_rgb = color.label2rgb(sp, img, kind='avg')
    plt.imshow(label_rgb)

    #----------- get labels ------------ 

    #labels = np.maximum(labels, rect)
    
    l = np.reshape(labels, (Y*X, 3))
    L = np.zeros((Y*X), dtype = int)

    for i in range(0,len(l)):
        a = "{:08b}".format(l[i,0])+"{:08b}".format(l[i,1])+"{:08b}".format(l[i,2])
        L[i] = int(a, 2)

    l = np.unique(L)
    num_labels = len(l)-1
    print('num labels: ' + str(num_labels))
    
    #----------- prepare train/test set ------------ 

    test_mask = np.zeros(N, dtype = 'bool')
    test_mask.fill(True)
    
    train_mask = np.zeros(N, dtype = 'bool')
    train_mask.fill(False)
    
    val_mask = np.zeros(N, dtype = 'bool')
    val_mask.fill(False)
    
    y = np.zeros(N, dtype = int)

    for i in range(0,N):
        y[i] = random.randint(0,num_labels-1)

    L = np.reshape(L, (X,Y))
    
    val = -1
    
    for lab in l:
        
        indx = np.where(lab == L)
        mask = np.zeros((X,Y), dtype = int)
        mask[indx] = 1
        mask = np.multiply(mask, sp)
        test = np.zeros((X,Y), dtype = int)
        
        if lab != bkgLabel:
            
            val = val + 1

            nl = np.unique(mask)   
            nl = np.delete(nl, 0)
        
            y[nl-1] = val
            train_mask[nl-1] = True
            test_mask[nl-1] = False
            #val_mask[nl-1] = False           

    test_mask = torch.from_numpy(np.asarray(test_mask)).bool()
    train_mask = torch.from_numpy(np.asarray(train_mask)).bool()
    val_mask = torch.from_numpy(np.asarray(val_mask)).bool()
    y = torch.from_numpy(np.asarray(y)).long()     
            
    data.test_mask = test_mask
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.y = y
    data.num_classes = num_labels
    
    print(data)
        

    #----------- train classfier ------------ 
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     
    #device = 'cpu'       
    
    model = Net()
    
    model, data = model.to(device), data.to(device)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.0005)
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
    patience = 1000
    counter = 0
        
    for epoch in range(1, 10001):
        
        train()
        train_acc = test()
        
        a = float(train_acc[0])
        
        if a > best_val_acc:
             best_val_acc = a
             counter = 0
        else:
             counter = counter + 1
        
        if counter >= patience:
            break
        
        if epoch % 100 == 0:
            print('File:' + tail + ', classes: ' +  str(len(l)-1)  + ', epoch: ' + str(epoch) +': acc: ' + str(train_acc))
        
        
    logits, accs = model(), []
    
    for _, mask in data('test_mask'):
        pred = logits[mask].max(1)[1]
    
    res = np.zeros(N)
    res[data.test_mask.cpu()] = pred.cpu()
    res[data.train_mask.cpu()] = data.y.cpu()[data.train_mask.cpu()]
    
    result = np.zeros((X, Y), dtype = int)
    
    for i in range(0, N):
        
        indx = np.where(sp == i + 1)
        result[indx] = res[i]
    
    plt.imshow(result)
    
    # result = np.zeros(Y*X)
    # result[data.test_mask.cpu()] = pred.cpu()
    # result[data.train_mask.cpu()] = data.y.cpu()[data.train_mask.cpu()]
    # result = np.reshape(result, [X,Y])
    # plt.imshow(result)
    
    #save result
    #result = 255*result/(len(l)-1)
    result = label2rgb(result)
    #path_result = dir_results + filename +'_beta_'+str('%.4f'%beta)+'.bmp'
    io.imsave(path_result, result.astype(int))
    
    io.imsave(path_result, result.astype(int))
    #result = np.zeros(321*481)
    #result[data.test_mask] = pred
    #result[data.train_mask] = data.y[data.train_mask]
    #result = np.reshape(result, [321, 481])
    #plt.imshow(result)
    
    #plt.imshow(labels)
    #plt.imshow(img)