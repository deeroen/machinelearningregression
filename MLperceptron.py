from sklearn.model_selection import GridSearchCV

from sklearn.neural_network import MLPRegressor
from data_treatment import *

#Parameters

param_grid = dict(learning_rate =['constant', 'invscaling', 'adaptive'])
#print (param_grid)
LR = MLPRegressor()


grid = GridSearchCV(LR, param_grid, cv = 5, scoring = M_squared_error)
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