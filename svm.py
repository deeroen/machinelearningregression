from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from data_treatment import *

from matplotlib import pyplot as plt


parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
svc = svm.SVR()
grid = GridSearchCV(svc, parameters, cv=3,scoring = M_squared_error)
grid.fit(tbl, np.array(Y1.astype(int).values))



print (grid.best_score_)
print (grid.best_params_)
print (grid.best_estimator_)


'''from sklearn.svm import SVR
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
clf = SVR(C=10, cache_size=200, coef0=0.0, degree=3, epsilon=0.1,
    gamma='auto_deprecated', kernel='linear', max_iter=-1, shrinking=True,
    tol=0.001, verbose=False)
clf.fit(tbl, np.array(Y1.astype(int).values))
pred = clf.predict(tbl)
rmse(pred,Y1.values.transpose())'''

