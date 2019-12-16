#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import sklearn.feature_selection as fs
from sklearn.feature_selection import SelectKBest
from sklearn.tree import DecisionTreeRegressor
from sklearn import feature_selection


# print des n features avec les plus grandes correlations
def print_correlation(X, Y, n):
    print("Correlation : ")
    corrcoef = np.corrcoef(X1_handled, X, Y, rowvar=False)[-1, :33]
    #print(np.around(corrcoef, decimals=3))
    most_corr = np.argsort(np.abs(corrcoef))
    feature_names = X1_handled.keys()
    for ind in most_corr[len(most_corr)-n: len(most_corr)]: # print des 10 meilleurs
        print("correlation = ", corrcoef[ind], '\t \t', ind, feature_names[ind])
    return most_corr
        

# print des n features avec les plus grandes mutual information
def print_mutual_information(X1_handled, X,Y, n):
    print("Mutual Information : ")
    mi = fs.mutual_info_regression(X, Y)
    #print(np.around(fs.mutual_info_regression(X, Y[:, 0]), 3))
    most_mi =  np.argsort(np.abs(mi))
    feature_names = X1_handled.keys()
    for ind in most_mi[len(most_mi)-n: len(most_mi)]:
        print( "mutual_information = ", mi[ind], '\t \t', ind, feature_names[ind])
    return most_mi

def features_selection(scaled_df, target, nb_of_features):
    # Select with mutual information
    selector = SelectKBest(feature_selection.mutual_info_regression, k=nb_of_features)
    selector.fit_transform(scaled_df, target)
    # Get columns to keep
    cols1 = scaled_df.columns[selector.get_support(indices=True)]

    # Select with F-value between label/feature for regression tasks
    selector = SelectKBest(feature_selection.f_regression, k=nb_of_features)
    selector.fit_transform(scaled_df, target)
    # Get columns to keep
    cols2 = scaled_df.columns[selector.get_support(indices=True)]

    # Select variables with tree
    clf = DecisionTreeRegressor(max_depth=5)
    clf = clf.fit(scaled_df, target)
    # Get the most important feature
    importances = clf.feature_importances_

    cols3 = list(scaled_df.columns[np.flip(np.argsort(importances)[-nb_of_features:])])

    features = list(set(list(cols1)) | set(list(cols2)) | set(list(cols3)))
    # Create new dataframe with only desired columns, or overwrite existing
    a_scaled = scaled_df[features]

    return a_scaled


