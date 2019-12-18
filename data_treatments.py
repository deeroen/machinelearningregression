import pandas as pd
import numpy as np
from load_data import *
from feature_selection import *
from sklearn.metrics import mean_squared_error, make_scorer
from math import sqrt
from sklearn.model_selection import train_test_split
from scipy.stats import shapiro

def custom_metric(y_test, y_pred):
    return sqrt(mean_squared_error(y_test, y_pred))
'''def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())'''
M_squared_error = make_scorer(custom_metric, greater_is_better=False)


'''Preprocessing of the data'''


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

# Create a new column for each station with binary variable
worktbl = pd.concat([worktbl, pd.get_dummies(X1['station']).add_prefix('station_')], axis=1)



'''Train-test split'''
X_train_valid, X_test, y_train_valid, y_test = train_test_split(worktbl, Y1, test_size=0.2, random_state=42)

#Scale the data
scaled_data = pd.DataFrame(preprocessing.scale(X_train_valid))
scaled_data.columns = worktbl.columns
scaled_data_test = pd.DataFrame(preprocessing.scale(X_test))
scaled_data_test.columns = worktbl.columns


mi = pd.DataFrame(fs.mutual_info_regression(scaled_data, y_train_valid))
mi.columns = np.array(['MI'])
mi['variable_name'] = scaled_data.columns
mi.sort_values(by=['MI'],inplace=True,ascending=False)
import numpy as np
import sklearn.feature_selection as fs
from sklearn.feature_selection import SelectKBest
from sklearn.tree import DecisionTreeRegressor
from sklearn import feature_selection

# Select variables with tree
clf = DecisionTreeRegressor(max_depth=3)
clf = clf.fit(scaled_data, y_train_valid)
# Get the most important feature
importances = pd.DataFrame(clf.feature_importances_)
importances.columns = np.array(['Tree_feature_importance'])
importances['variable_name'] = scaled_data.columns
importances.sort_values(by=['Tree_feature_importance'],inplace=True,ascending=False)

corrcoef = pd.DataFrame(abs(np.corrcoef(scaled_data, y_train_valid, rowvar=False)[-1, :34]))
# Get the most important feature
corrcoef.columns = np.array(['Corrcoef'])
corrcoef['variable_name'] = scaled_data.columns
corrcoef.sort_values(by=['Corrcoef'],inplace=True,ascending=False)


n= 10
t0 = pd.merge(corrcoef.head(n),importances.head(n), on=['variable_name'],how='outer')
t1 = pd.merge(t0,mi.head(n), on=['variable_name'],how='outer')

t3 = pd.DataFrame(t1['variable_name'])
t3['Mutual Information'] = pd.Series(t1['MI']).notna()
t3['Tree_feature_importance'] = pd.Series(t1['Tree_feature_importance']).notna()
t3['Corrcoef'] = pd.Series(t1['Corrcoef']).notna()
'''tbl = features_selection(X_train_valid, y_train_valid, 7)'''

print(t3.to_latex(index=False))



from sklearn import tree
import pydotplus
# Create DOT data
dot_data = tree.export_graphviz(clf,impurity=False, out_file=None,
                                feature_names=list(scaled_data.columns),filled=True,
                                rounded=True,
                                special_characters=True)
# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png('C:\\Users\cocol\PycharmProjects\machinelearningregression\\tree.png')

'''This funciton takes a imput a dataframe and return yes if all the features are distributed normally and false otherwise'''


def isnormal(data):
    # Shapiro-Wilk Test
    for i in range(0,data.shape[1]):
        # normality test
        _, p = shapiro(data.iloc[:,i])
        # interpret
        alpha = 0.05
        list = []
        if p <= alpha:
            print(p)
            list.append(data.columns[i])
            print('Sample does not look Gaussian (reject H0) at least for feature: '+str(data.columns[i]))

    return list
isnormal(worktbl)
import matplotlib.pyplot as plt
