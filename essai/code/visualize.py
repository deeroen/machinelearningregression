#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt

def visualize(X, Y, name, save):
    n_sample, n_feature = X.shape
    feature_names = X.keys()
    
    #plt.figure(dpi=400)
    for k in feature_names :
        plt.figure(dpi=200)
        plt.scatter(X[k], Y)
        plt.title(k)
        #plt.show()
        if(save):
            plt.savefig(name+'_'+str(k)+'.png')

def visualize2(X, name, save):
    n_sample, n_feature = X.shape
    feature_names = X.keys()
    
    for k in feature_names :
        plt.figure(dpi=200)
        plt.scatter(range(len(X[k])), X[k])
        plt.title(k)
        #plt.show()
        if(save):
            plt.savefig(name+'_'+str(k)+'.png')

