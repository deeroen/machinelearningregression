from sklearn.model_selection import GridSearchCV

from data_treatment import *
from sklearn.tree import DecisionTreeRegressor
#Parameters

param_grid = p_grid = {"max_depth": [2,3,6,10,15,20],"min_samples_split" : [5,8,10,15,20],"min_impurity_decrease" : [0,0.01],'criterion':['mse','friedman_mse','mae']}
#print (param_grid)
Tree = DecisionTreeRegressor()

grid = GridSearchCV(Tree, param_grid, cv = 3, scoring = M_squared_error)
grid.fit(tbl,Y1)








print (grid.best_score_)
print (grid.best_params_)
print (grid.best_estimator_)


'''def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
clf = DecisionTreeRegressor(criterion='mae', max_depth=6, max_features=None,
                      max_leaf_nodes=None, min_impurity_decrease=0.01,
                      min_impurity_split=None, min_samples_leaf=1,
                      min_samples_split=20, min_weight_fraction_leaf=0.0,
                      presort=False, random_state=None, splitter='best')
clf.fit(tbl, Y1)
pred = clf.predict(tbl)
rmse(pred,Y1.values)'''
