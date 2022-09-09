# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 10:42:34 2019

@author: an_fab
"""

import os
import glob

import numpy as np
import matplotlib.pyplot as plt

from skimage import io
from skimage.exposure import rescale_intensity
from skimage.segmentation import random_walker

from helpers import show_img
   
#----------- some settings -------------

bkgLabel = int("{:08b}".format(0)+"{:08b}".format(0)+"{:08b}".format(0), 2)

#----------- data paths ------------

main_dir = 'C:\\Users\\an_fab\\Desktop\\graph_segm\\microsoft'

dir_gt = 'C:\\Users\\an_fab\\Desktop\\graph_segm\\microsoft'+'\\gt\\'
dir_img = 'C:\\Users\\an_fab\\Desktop\\graph_segm\\microsoft'+'\\org\\'
dir_results = 'C:\\Users\\an_fab\\Desktop\\graph_segm\\microsoft'+'\\results_random_walker\\'
dir_scrib = 'C:\\Users\\an_fab\\Desktop\\graph_segm\\microsoft'+'\\scribbles\\'

for name in glob.glob(dir_img+'/*'):
    
    path_img = name
    
    head, tail = os.path.split(name)
    filename, file_extension = os.path.splitext(tail)
    path_labels = dir_scrib + filename +'.bmp'
    path_gt = dir_gt + filename +'.bmp'
    path_result = dir_results + filename +'.bmp'

    #----------- read inputs ------------

    print('Processing: ' + path_img)
    img = io.imread(path_img)
    labels = io.imread(path_labels)
    gt = io.imread(path_gt)

    # show_img(img)
    # show_img(labels)
    
    Y, X = img.shape[1], img.shape[0]
    c = 3 if len(img.shape)==3 else 1
    
    l = np.reshape(labels, (Y*X, 3))
    L = np.zeros((Y*X), dtype = int)
    markers = np.zeros((X, Y), dtype = int)

    for i in range(0,len(l)):
        a = "{:08b}".format(l[i,0])+"{:08b}".format(l[i,1])+"{:08b}".format(l[i,2])
        L[i] = int(a, 2)

    lu = np.unique(L)
    print('num labels: ' + str(len(lu)-1))
    
    L = np.reshape(L, (X,Y))
    
    m = 0
    
    for lab in lu:
        
        indx = np.where(lab == L)
        
        if lab == bkgLabel:
            markers[indx] = 0
        else:
            m = m + 1
            markers[indx] = m
    
    
    result = random_walker(img, L, beta=5000, multichannel=True, mode='bf')
    result = rescale_intensity(result,(0, m),(0,255))
    plt.imshow(result)

    io.imsave(path_result, result.astype(np.uint8))
