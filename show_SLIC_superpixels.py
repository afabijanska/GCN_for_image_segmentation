# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 11:42:17 2020

@author: an_fab
"""

import numpy as np
import matplotlib.pyplot as plt

from skimage import io
from skimage.color import label2rgb
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.future import graph

path = 'test5.jpg'

img = io.imread(path)
plt.imshow(img)

Y, X = img.shape[1], img.shape[0]
N = int(0.01*Y*X)
   
sp = slic(img, n_segments=N, sigma = 2, convert2lab=True)
sp = sp + 1
g = graph.rag_mean_color(img, sp)

plt.imshow(label2rgb(sp))
plt.imshow(mark_boundaries(img, sp))
io.imsave('spx5.bmp',label2rgb(sp))
io.imsave('spx5_B.bmp',mark_boundaries(img, sp))

# fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(6, 8))

# ax[0].set_title('RAG drawn with default settings')
# lc = graph.show_rag(sp, g, img, ax=ax[0])
# # specify the fraction of the plot area that will be used to draw the colorbar
# fig.colorbar(lc, fraction=0.03, ax=ax[0])
