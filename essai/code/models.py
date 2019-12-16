#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


from competitive_learning import *
from linear_model import *
from rbfn import *


# rbfn vu en TP
def rbfn_tp(X_train, Y_train, X_test, Y_test, n_center, smooth_f):
    
    def score_rbfn(n_center, smooth_f, X, T, X_test, T_test):
    rbfn = MyRBFN(n_center, smooth_f)
    rbfn.fit(X, T)
    return rbfn.score(X_test, T_test)
    
    # split train-validation set
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)
    rmses = {}
    best = {'rmse':1000}

    for nc in n_center:
        rmses[nc] = list()

        for sf in smooth_f:
            rmse = score_rbfn(nc, sf, X1_train_selected, Y1_train, X1_valid_selected, Y1_valid)
            rmses[nc].append(rmse)
            if rmse < best['rmse']:
                best['rmse'] = rmse
                best['n_center'] = nc
                best['smooth_f']= sf
    
    rmse = score(best['n_center'], best['smooth_f'], X_train, Y_train, X_test, Y_Test)
    return rmse
 
# linear regression vue en tp
def linear_regression_tp(X, Y, X_test, Y_test):
    linear_model = MyLinearRegressor(add_bias = True)
    linear_model.fit(X, Y)
    return linear_model.score(X_test, Y_test)



def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
def custom_metric(y_test, y_pred):
    return sqrt(mean_squared_error(y_test, y_pred))
M_squared_error = make_scorer(custom_metric, greater_is_better=False)


def KNN(X_train, Y_train, X_test, Y_test, k_range=list(range(1,31)) + [50,100], weight_options = ["uniform", "distance"]):
    
    param_grid = dict(n_neighbors = k_range, weights = weight_options)
    knn = KNeighborsRegressor()
    
    grid = GridSearchCV(knn, param_grid, cv = 5, scoring = M_squared_error)
    grid.fit(X_train,Y_train)

    #print (grid.best_score_)
    #print (grid.best_params_)
    #print (grid.best_estimator_)

    clf = grid.best_estimator_
    clf.fit(X_train,Y_train)

    pred = clf.predict(X_test[tbl.columns])
    return rmse(pred,y_test.values)
    
def linear_regression(X_train, Y_train, X_test, Y_test):
  
    param_grid = dict(fit_intercept = [True]) 
    lr = LinearRegression().fit(X_train,Y_train)
    pred = lr.predict(X_test[tbl.columns])
    return rmse(pred,Y_test.values)
    
def tree(X_train, Y_train, X_test, Y_test ):
    
    param_grid = {"max_depth": [2,3,5,10,15],"min_samples_split" : [5,10,20],"min_impurity_decrease" : [0,0.01,0.02,0.1],'criterion':['mse','friedman_mse','mae']}
    Tree = DecisionTreeRegressor()

    grid = GridSearchCV(Tree, param_grid, cv = 5, scoring = M_squared_error)
    grid.fit(X_train,Y_train)

    #print (grid.best_score_)
    #print (grid.best_params_)
    #print (grid.best_estimator_)

    clf = grid.best_estimator_
    clf.fit(X_train,Y_train)

    pred = clf.predict(X_test[tbl.columns])
    return rmse(pred,Y_test.values)
    
def random_forest():
    print('todo')

def MLperceptron():
    print('todo')
    
def svm():
    print('todo')

