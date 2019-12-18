from load_data import *
import pandas as pd

hist = Y1.hist(bins=300)

tbl = X1.loc[Y1['Label']<400]
Y1 = Y1[Y1['Label']<400]