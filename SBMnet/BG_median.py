#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import os.path
import cv2
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random
import re
from numpy.linalg import matrix_rank
from numpy.linalg import qr
from numpy.linalg import svd
from sklearn.preprocessing import StandardScaler

global dim
dim = 1

def BG_median(video_path):
    Frames = [None] * 10000
    nFrames = 0
    
    for dirpath, dirnames, filenames in os.walk(video_path):
        for filename in [f for f in filenames if f.endswith(".jpg")]:
            image = cv2.imread(os.path.join(dirpath, filename),-1)
            x = re.findall(r"in[0]+([1-9][0-9]*)\.jpg", filename)
            if len(x)==0:
                Frames[0]=image
                continue
            Frames[int(x[0])]=image
            nFrames = nFrames if (nFrames>int(x[0])) else int(x[0])
    
    Frames = np.array(Frames[0:nFrames])
    if nFrames>5000:
        Frames = Frames[range(0,nFrames,10)]
        nFrames = int(nFrames/10)
    else:
        if nFrames > 2000:
            Frames = Frames[range(0,nFrames,5)]
            nFrames = int(nFrames/5)
        else:
            if nFrames > 1000:
               Frames = Frames[range(0,nFrames,2)]
               nFrames = int(nFrames/2)


    dim = Frames.shape[1]*Frames.shape[2]*Frames.shape[3]
    
    
    X = np.reshape(Frames, (-1,dim))
    
    pca = PCA(n_components=5)
    pca.fit(X)
    Y = pca.inverse_transform(pca.transform(X))
    Y = np.reshape(Y,Frames.shape[0:4])
    Y = np.median(Y, axis=0)
    Y = np.around(Y)
    Y = Y.astype('uint8')
    return Y

def gaussian_kernel(X):
    Gram = np.matmul(X, np.transpose(X))/dim
    col = np.reshape(np.sum(X*X,axis=1)/dim, (-1,1))*5
    row = np.transpose(col)
    return np.exp(10*Gram-col-row)

def variance_kernel(X):
    Gram = np.matmul(X, np.transpose(X))/dim
    return 1+Gram

def skewness_kernel(X):
    Gram = np.matmul(X, np.transpose(X))/dim
    return 1+Gram+Gram*Gram

def kurtosis_kernel(X):
    Gram = np.matmul(X, np.transpose(X))/dim
    GG=Gram*Gram
    return 1+Gram+GG+GG*Gram

def kernel_calc(X, kernel_function, n_components):
    N = X.shape[0]
    k = X.shape[1]
    scaler = StandardScaler().fit(X)
    X=scaler.transform(X)
    q, r = np.linalg.qr(np.transpose(X), mode='reduced')
    M=kernel_function(X)
    a=np.matmul(r,np.matmul(M,np.transpose(r)))
    U,O,V = np.linalg.svd(a, full_matrices=True)
    PC = np.matmul(q,U)
    
    ret = np.zeros((N,k))
    for i in range(n_components):
        b = np.reshape(PC[:,i],(k,1))
        ret = ret+np.matmul(np.matmul(X,b),np.transpose(b))
    ret = scaler.inverse_transform(ret)
    return ret

def var_median(video_path):
    Frames = [None] * 10000
    nFrames = 0
    
    for dirpath, dirnames, filenames in os.walk(video_path):
        for filename in [f for f in filenames if f.endswith(".jpg")]:
            image = cv2.imread(os.path.join(dirpath, filename),-1)
            x = re.findall(r"in[0]+([1-9][0-9]*)\.jpg", filename)
            if len(x)==0:
                Frames[0]=image
                continue
            Frames[int(x[0])]=image
            nFrames = nFrames if (nFrames>int(x[0])) else int(x[0])
    
    Frames = np.array(Frames[0:nFrames])
    dim = Frames.shape[1]*Frames.shape[2]*Frames.shape[3]
    
    
    X = np.reshape(Frames, (-1,dim))
    Y = kernel_calc(X,variance_kernel,n_components=5)    
    Y = np.reshape(Y,Frames.shape[0:4])
    Y = np.median(Y, axis=0)
    Y = np.around(Y)
    Y = Y.astype('uint8')
    return Y

def skew_median(video_path):
    Frames = [None] * 10000
    nFrames = 0
    
    for dirpath, dirnames, filenames in os.walk(video_path):
        for filename in [f for f in filenames if f.endswith(".jpg")]:
            image = cv2.imread(os.path.join(dirpath, filename),-1)
            x = re.findall(r"in[0]+([1-9][0-9]*)\.jpg", filename)
            if len(x)==0:
                Frames[0]=image
                continue
            Frames[int(x[0])]=image
            nFrames = nFrames if (nFrames>int(x[0])) else int(x[0])
    
    Frames = np.array(Frames[0:nFrames])
    dim = Frames.shape[1]*Frames.shape[2]*Frames.shape[3]
    
    
    X = np.reshape(Frames, (-1,dim))
    Y = kernel_calc(X,skewness_kernel,n_components=5)    
    Y = np.reshape(Y,Frames.shape[0:4])
    Y = np.median(Y, axis=0)
    Y = np.around(Y)
    Y = Y.astype('uint8')
    return Y

def kurtosis_median(video_path):
    Frames = [None] * 10000
    nFrames = 0
    
    for dirpath, dirnames, filenames in os.walk(video_path):
        for filename in [f for f in filenames if f.endswith(".jpg")]:
            image = cv2.imread(os.path.join(dirpath, filename),-1)
            x = re.findall(r"in[0]+([1-9][0-9]*)\.jpg", filename)
            if len(x)==0:
                Frames[0]=image
                continue
            Frames[int(x[0])]=image
            nFrames = nFrames if (nFrames>int(x[0])) else int(x[0])
    
    Frames = np.array(Frames[0:nFrames])
    dim = Frames.shape[1]*Frames.shape[2]*Frames.shape[3]
    
    
    X = np.reshape(Frames, (-1,dim))
    Y = kernel_calc(X,kurtosis_kernel,n_components=5)    
    Y = np.reshape(Y,Frames.shape[0:4])
    Y = np.median(Y, axis=0)
    Y = np.around(Y)
    Y = Y.astype('uint8')
    return Y

def gaussian_median(video_path):
    Frames = [None] * 10000
    nFrames = 0

    for dirpath, dirnames, filenames in os.walk(video_path):
        for filename in [f for f in filenames if f.endswith(".jpg")]:
            image = cv2.imread(os.path.join(dirpath, filename),-1)
            x = re.findall(r"in[0]+([1-9][0-9]*)\.jpg", filename)
            if len(x)==0:
                Frames[0]=image
                continue
            Frames[int(x[0])]=image
            nFrames = nFrames if (nFrames>int(x[0])) else int(x[0])

    Frames = np.array(Frames[0:nFrames])
    dim = Frames.shape[1]*Frames.shape[2]*Frames.shape[3]


    X = np.reshape(Frames, (-1,dim))
    Y = kernel_calc(X,gaussian_kernel,n_components=5)
    Y = np.reshape(Y,Frames.shape[0:4])
    Y = np.median(Y, axis=0)
    Y = np.around(Y)
    Y = Y.astype('uint8')
    return Y
