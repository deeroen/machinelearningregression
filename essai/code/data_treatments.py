#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import RobustScaler

from scipy.stats import shapiro
def handlecyclic(data_frame):
    df = data_frame.copy()
    # handle hour values
    df['hr_sin'] = np.sin(df['hour']*(2.*np.pi/24))
    df['hr_cos'] = np.cos(df['hour']*(2.*np.pi/24))
    # handle month values
    df['mnth_sin'] = np.sin((df['month']-1)*(2.*np.pi/12))
    df['mnth_cos'] = np.cos((df['month']-1)*(2.*np.pi/12))
    # handle day values
    df['day_sin'] = np.sin((df['day'] - 1) * (2. * np.pi / 31))
    df['day_cos'] = np.cos((df['day'] - 1) * (2. * np.pi / 31))
    # handle wind direction
    df = df.replace({'N': 0,'NNE':1,'NE':2,'ENE':3,'E':4,'ESE':5,'SE':6,'SSE':7,'S':8,'SSW':9,'SW':10,'WSW':11,'W':12,'WNW':13,'NW':14,'NNW':15})
    df['wd_sin'] = np.sin((df['wd']) * (2. * np.pi / 16))
    df['wd_cos'] = np.cos((df['wd']) * (2. * np.pi / 16))
    df = df.drop(['wd'], axis=1)
    return df

def handle_station(df):
    return pd.concat([df.drop(['station'], axis=1), pd.get_dummies(df['station']).add_prefix('station_')], axis=1)

def add_linear_time(data_frame):
    df2 = pd.DataFrame()
    df2['time'] = ((data_frame['year']+data_frame['month']/12)*100+data_frame['day']/31)*100+data_frame['hour']/24
    #df2['time'] = data_frame['year']*1000000+(data_frame['month']/12)*1000+(data_frame['day']/31)*10+(data_frame['hour']/24)
    return pd.concat([data_frame, df2], axis=1)

def stand(X, X_test, p):
    # Z-score standardisation
    std_scaler = StandardScaler().fit(X)
    
    if p:
        print(std_scaler.mean_)
        print(std_scaler.scale_)
    
    X_stand = std_scaler.transform(X)
    X_test_stand = std_scaler.transform(X_test)
    return [X_stand, X_test_stand]


def robust_scaler(X, X_test):
    transformer = RobustScaler().fit(X)
    X_trans = transformer.transform(X)
    X_test_trans = transformer.transform(X_test)
    return [X_trans, X_test_trans]
    

'''This funciton takes as imput a dataframe and return a list of all the features that are not distributed normally'''
def isnormal(data):
    # Shapiro-Wilk Test
    for i in range(0,data.shape[1]):
        # normality test
        _, p = shapiro(data.iloc[:,i])
        # interpret
        alpha = 0.05
        list = []
        if p <= alpha:
            print(p)
            list.append(data.columns[i])
            print('Sample does not look Gaussian (reject H0) at least for feature: '+str(data.columns[i]))
    return list



