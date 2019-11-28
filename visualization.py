from load_data import *

import scipy.io
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib as mpl


n_samples_X1, n_feats_X1 = X1.shape
n_samples_X2, n_feats_X2 = X2.shape

feature_names = ["year", "month", "day", "hour", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "wd", "WSPM", "station", "PM2.5"]

#Y1 = np.array(Y1)
Y1 = Y1.to_numpy()
X1 = X1.to_numpy()

fig, axs = plt.subplots(nrows=5,ncols=3)
fig.suptitle('Content of ')

axs[0,0].scatter(X1[:,5], Y1)
axs[0,0].set_title('year')


axs[0,1].scatter(X1[:,6], Y1)
axs[0,1].set_title('month')

#plt.rcParams["figure.figsize"] = (40,20) # remove to see overlapping subplots
plt.show()



#fig=plt.figure(figsize=(10, 10), dpi=90)
#for i in range(4,15):
#    plt.subplot(n_feats_X1 // 3 + 1, 3, i + 1)
#    plt.scatter(X1[:, i], Y1, s=10)
#    plt.title(f"{t_names[i]}")
#plt.tight_layout()
#plt.show()


#corr_vec = np.corrcoef(X1,Y1, rowvar=False)[-1, :n_feats_X1]
#most_corr = np.argsort(np.abs(corr_vec))

#print(corr_vec)
#print(most_corr)

    # If rowvar is True (default), then each row represents a variable, with observations in the columns. Otherwise, the relationship is transposed: each column represents a variable, while the rows contain observations.
    # argsort : Returns the indices that would sort an array.

#fig=plt.figure(figsize=(10, 10), dpi=90)
#for i, ind in enumerate(most_corr):
#    plt.subplot(n_feats_X1//3+1,3,i+1)
#    plt.scatter(X1[:,ind], Y1, s=10) # (s : The marker size in points**2)
#    plt.title(f"{t_names[ind]}: {corr_vec[ind]:.4}")
#plt.tight_layout()
#plt.show()
