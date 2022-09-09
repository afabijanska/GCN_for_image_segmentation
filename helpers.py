# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 14:45:10 2020

@author: an_fab
"""

import matplotlib.pyplot as plt

def show_img(img):
    width = 10.0
    height = img.shape[0]*width/img.shape[1]
    f = plt.figure(figsize=(width, height))
    plt.imshow(img)

def sub2ind(array_shape, rows, cols):
    return rows*array_shape[1] + cols

def ind2sub(array_shape, ind):
    rows = int(int(ind) / array_shape[1])
    cols = (int(ind) % array_shape[1])
    return (rows, cols)