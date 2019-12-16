from sklearn.model_selection import GridSearchCV

from data_treatments import *
from sklearn.tree import DecisionTreeRegressor
#Parameters

param_grid = p_grid = {"max_depth": [2,3,5,10,15],"min_samples_split" : [5,10,20],"min_impurity_decrease" : [0,0.01,0.02,0.1],'criterion':['mse','friedman_mse','mae']}
#print (param_grid)
Tree = DecisionTreeRegressor()

grid = GridSearchCV(Tree, param_grid, cv = 5, scoring = M_squared_error)
grid.fit(tbl,y_train_valid)


print (grid.best_score_)
print (grid.best_params_)
print (grid.best_estimator_)


print('Results on the test set')

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
clf = grid.best_estimator_
clf.fit(tbl,y_train_valid)

pred = clf.predict(X_test[tbl.columns])
print(rmse(pred,y_test.values))