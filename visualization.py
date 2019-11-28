from load_data import *

import scipy.io
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib as mpl


n_samples_X1, n_feats_X1 = X1.shape
n_samples_X2, n_feats_X2 = X2.shape

feature_names = ["year", "month", "day", "hour", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "wd", "WSPM", "station", "PM2.5"]

X1 = X1.to_numpy()
Y1 = Y1.to_numpy()

fig=plt.figure(figsize=(10, 10), dpi=90)
for i in range(1,15):
    plt.subplot(n_feats_X1 // 3 + 1, 3, i + 1)
    plt.scatter(X1[:, i], Y1, s=10)
    plt.title(f"{feature_names[i]}")
plt.tight_layout()
plt.show()


