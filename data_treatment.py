import pandas as pd
import numpy as np
from feature_selection import features_selection
from load_data import *
#worktbl = pd.concat([X1.drop(['wd'], axis=1), pd.get_dummies(X1['wd'])], axis=1)
# Handling ciclic variables
# http://blog.davidkaleko.com/feature-engineering-cyclical-features.html

def handlecyclic(data_frame):
    df = data_frame.copy()
    df['hr_sin'] = np.sin(df['hour']*(2.*np.pi/24))
    df['hr_cos'] = np.cos(df['hour']*(2.*np.pi/24))
    df['mnth_sin'] = np.sin((df['month']-1)*(2.*np.pi/12))
    df['mnth_cos'] = np.cos((df['month']-1)*(2.*np.pi/12))
    df['day_sin'] = np.sin((df['day'] - 1) * (2. * np.pi / 12))
    df['day_cos'] = np.cos((df['day'] - 1) * (2. * np.pi / 12))
    df = df.replace({'N': 0,'NNE':1,'NE':2,'ENE':3,'E':4,'ESE':5,'SE':6,'SSE':7,'S':8,'SSW':9,'SW':10,'WSW':11,'W':12,'WNW':13,'NW':14,'NNW':15})
    df['wd_sin'] = np.sin((df['wd']) * (2. * np.pi / 16))
    df['wd_cos'] = np.cos((df['wd']) * (2. * np.pi / 16))
    df = df.drop(['wd'], axis=1)
    return df

worktbl = handlecyclic(X1)

worktbl = pd.concat([worktbl.drop(['station'], axis=1), pd.get_dummies(X1['station']).add_prefix('station_')], axis=1)
tbl = features_selection(worktbl, Y1, 7)