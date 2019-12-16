from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from data_treatments import *
# Build a classification task using 3 informative features




param_grid = {'n_estimators': [500, 700, 1000], 'max_depth': [1, 2, 3], 'min_samples_split': [2, 3]}

grid = clf = GridSearchCV(RandomForestRegressor(), param_grid, cv=5,  n_jobs=1, verbose=1)
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