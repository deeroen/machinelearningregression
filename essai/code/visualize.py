#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt

dico = {'year':'year',
       'month':'month',
       'day':'day',
       'hour':'hour',
       'SO2':'SO2 concentration (ug/m3)',
       'NO2':'NO2 concentration (ug/m3)',
       'CO':'CO concentration (ug/m3)',
       'O3':'O3 concentration (ug/m3)',
       'TEMP':'temperature',
       'PRES':' pressure (hPa)',
       'DEWP':'dew point temperature (degree Celsius)',
       'RAIN':'precipitation (mm)',
       'wd':'wind direction',
       'WSPM':'wind speed (m/s)',
       'station':'id of the air-quality monitoring site',
       'PM2.5': 'PM2.5 concentration (ug/m3)',
       'hr_sin':'hr_sin',
       'hr_cos':'hr_cos',
       'mnth_sin':'mnth_sin',
       'mnth_cos':'mnth_cos',
       'day_sin':'day_sin',
       'day_cos':'day_cos',
       'wd_sin':'wd_sin',
       'wd_cos':'wd_cos',
       'time':'time'}

        
        
def visualize(X, Y, name, save):
    n_sample, n_feature = X.shape
    feature_names = X.keys()
    
    for k in feature_names :
        plt.figure(dpi=200)
        plt.scatter(X[k], Y)
        plt.xlabel(dico[k])
        plt.ylabel(dico['PM2.5'])
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

