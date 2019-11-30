from sklearn.feature_selection import SelectKBest
import sklearn.feature_selection as fs
from sklearn.feature_selection import chi2
from sklearn import feature_selection, tree
import pandas as pd
from sklearn import preprocessing
import numpy as np

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


def features_selection(features_df, target, nb_of_features):
    # normalize the table
    min_max_scaler = preprocessing.MinMaxScaler()
    scaled_df = pd.DataFrame(min_max_scaler.fit_transform(features_df))
    scaled_df.columns = features_df.columns

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


    features = list(set(list(cols1)) | set(list(cols2)))
    # Create new dataframe with only desired columns, or overwrite existing
    a_scaled = scaled_df[features]

    return a_scaled