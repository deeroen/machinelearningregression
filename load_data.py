import pandas as pd
X1 = pd.read_csv("Datasets-20191122\X1.csv")
Y1 = pd.read_csv("Datasets-20191122\Y1.csv",sep='\t',names=["Label"])
X2 = pd.read_csv("Datasets-20191122\X2.csv")

from feature_selection import features_selection

worktbl = pd.concat([X1.drop(['wd'], axis=1), pd.get_dummies(X1['wd'])], axis=1)

tbl = features_selection(worktbl, Y1, 7)