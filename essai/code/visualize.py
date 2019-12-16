#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt

def visualize(X, Y, name, save):
    n_sample, n_feature = X.shape
    feature_names = X.keys()
    
    for k in feature_names :
        plt.scatter(X[k], Y)
        plt.title(k)
        plt.show()
    if(save):
        plt.savefig(name)

