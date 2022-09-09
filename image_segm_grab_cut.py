# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 10:42:34 2019

@author: an_fab
"""

import os
import glob

import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

from skimage import io
from skimage.exposure import rescale_intensity

from helpers import show_img
   
#----------- some settings -------------

bkgLabel = int("{:08b}".format(0)+"{:08b}".format(0)+"{:08b}".format(0), 2)
objLabel = int("{:08b}".format(255)+"{:08b}".format(0)+"{:08b}".format(0), 2)

#----------- data paths ------------

main_dir = 'C:\\Users\\an_fab\\Desktop\\graph_segm\\microsoft'

dir_gt = 'C:\\Users\\an_fab\\Desktop\\graph_segm\\microsoft'+'\\gt\\'
dir_img = 'C:\\Users\\an_fab\\Desktop\\graph_segm\\microsoft'+'\\org\\'
dir_results = 'C:\\Users\\an_fab\\Desktop\\graph_segm\\microsoft'+'\\results_grabcut\\'
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

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    # show_img(img)
    # show_img(labels)
    
    Y, X = img.shape[1], img.shape[0]
    c = 3 if len(img.shape)==3 else 1
    
    l = np.reshape(labels, (Y*X, 3))
    L = np.zeros((Y*X), dtype = int)
    markers = np.empty((X, Y), dtype = np.uint8)

    for i in range(0,len(l)):
        a = "{:08b}".format(l[i,0])+"{:08b}".format(l[i,1])+"{:08b}".format(l[i,2])
        L[i] = int(a, 2)

    lu = np.unique(L)
    print('num labels: ' + str(len(lu)-1))
    
    L = np.reshape(L, (X,Y))
    
    m = 0
    
    for lab in lu:
        m = m + 1
        indx = np.where(lab == L)
        
        if lab == bkgLabel:
            markers[indx] = random.randint(2,3)
            #markers[indx] = 3
        elif lab == objLabel:
            markers[indx] = 1
        else:
            markers[indx] = 0
            
            

    
    rect = (0,0,X-1,Y-1)
    cv2.grabCut(img,markers,[],bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
    
    result = markers
    result[np.where(result == 2)] = 0
    result[np.where(result == 3)] = 1
    #result = random_walker(img, L, beta=5000, multichannel=True, mode='bf')
    result = rescale_intensity(result,(0, 1),(0,255))
    #plt.imshow(result)

    io.imsave(path_result, result.astype(np.uint8))
