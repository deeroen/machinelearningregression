from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression
from data_treatment import *

#Parameters

param_grid = dict(fit_intercept = [True])
#print (param_grid)
LR = LinearRegression()

grid = GridSearchCV(LR, param_grid, cv = 3, scoring = M_squared_error)
grid.fit(tbl,Y1)








print (grid.best_score_)
print (grid.best_params_)
print (grid.best_estimator_)


'''def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
clf = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
clf.fit(tbl, Y1)
pred = clf.predict(tbl)
rmse(pred,Y1)'''
