import sys
working_dirr = "C:\\Users\cocol\Desktop\Q1_ingé_civil_M2\Machine learning\project"
sys.path.append(working_dirr)
import pandas as pd
X1 = pd.read_csv(working_dirr+"\Datasets-20191122\X1.csv")
Y1 = pd.read_csv(working_dirr+"\Datasets-20191122\Y1.csv")
X2 = pd.read_csv(working_dirr+"\Datasets-20191122\X2.csv")
Y1bis = Y1.append({"3.0":123}, ignore_index=True)
from feature_selection import features_selection

worktbl = pd.concat([X1.drop(['wd'], axis=1), pd.get_dummies(X1['wd'])], axis=1)

tbl = features_selection(worktbl, Y1bis, 7)