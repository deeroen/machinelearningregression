from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from data_treatment import *
# Build a classification task using 3 informative features




param_grid = {'n_estimators': [500, 700, 1000], 'max_depth': [1, 2, 3], 'min_samples_split': [2, 3]}

grid = clf = GridSearchCV(RandomForestRegressor(), param_grid, cv=3,  n_jobs=1, verbose=1)
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