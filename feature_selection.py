from sklearn.feature_selection import SelectKBest
import sklearn.feature_selection as fs
from sklearn.feature_selection import chi2
from sklearn import feature_selection
import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing

def print_correlation(X, Y):
    print("Correlation : ")
    corrcoef = np.corrcoef(X, Y, rowvar=False)[-1, :33]
    #print(np.around(corrcoef, decimals=3))
    most_corr = np.argsort(np.abs(corrcoef))
    #print(most_corr)
    feature_names = list(X)
    for ind in most_corr:
        print(ind, feature_names[ind], "correlation = ", corrcoef[ind])

def print_mutual_information(X,Y):
    print("Mutual Information : ")
    mi = fs.mutual_info_regression(X.to_numpy(), Y.to_numpy()[:, 0])
    #print(np.around(fs.mutual_info_regression(X, Y[:, 0]), 3))
    most_mi =  np.argsort(np.abs(mi))
    feature_names = list(X)
    for ind in most_mi:
        print(ind, feature_names[ind], "mutual_information = ", mi[ind])


def features_selection(scaled_df, target, nb_of_features):



    # Select with mutual information
    selector = SelectKBest(feature_selection.mutual_info_regression, k=nb_of_features)
    selector.fit_transform(scaled_df, target)
    # Get columns to keep
    cols1 = scaled_df.columns[selector.get_support(indices=True)]

    # Select with corrcoef between label/feature for regression tasks
    corrcoef = pd.DataFrame(abs(np.corrcoef(scaled_df, target, rowvar=False)[-1, :scaled_df.shape[1]]))
    # Get the most important feature
    corrcoef.columns = np.array(['Corrcoef'])
    corrcoef['variable_name'] = scaled_df.columns
    corrcoef.sort_values(by=['Corrcoef'], inplace=True, ascending=False)
    cols2 = corrcoef['variable_name'].head(nb_of_features)
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