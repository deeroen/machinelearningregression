#https://gist.github.com/sawansaurabh/3748a083ffdec38aacb8f43660a5f654
from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsRegressor
from data_treatments import *





k_range = list(range(1,31)) + [50,100]
weight_options = ["uniform", "distance"]

param_grid = dict(n_neighbors = k_range, weights = weight_options)
#print (param_grid)
knn = KNeighborsRegressor()

grid = GridSearchCV(knn, param_grid, cv = 5, scoring = M_squared_error)
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