from data_treatments import *

import numpy as np
from sklearn.decomposition import PCA

pca = PCA(n_components=5)
pca.fit(worktbl)


print(pca.explained_variance_ratio_)

print(pca.singular_values_)