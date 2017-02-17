#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 21:59:11 2017

@author: suman
"""
import cPickle, gzip,pickle
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
def elasticDeform(im,alpha=34,sigma=4):
    im=np.reshape(im,(28,28))
    m=im.shape
    #set up the displacement field
    dx=gaussian_filter((np.random.rand(*m)*2-1),sigma)*alpha
    dy=gaussian_filter((np.random.rand(*m)*2-1),sigma)*alpha
    #set up co-ordinates for displacement
    x,y=np.meshgrid(np.arange(m[0]), np.arange(m[1]))
    i=np.reshape(y+dy,(-1,1)),np.reshape(x+dx,(-1,1))
    transformed=map_coordinates(im,i,order=1).reshape(m).flatten()
    return transformed
f = gzip.open('data/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()
distorted_data=np.apply_along_axis(elasticDeform,axis=1,arr=train_set[0])
f=open('data/mnist_distorted.pkl','w+')
pickle.dump([distorted_data],f)
f.close()