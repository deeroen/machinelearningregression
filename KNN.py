#https://gist.github.com/sawansaurabh/3748a083ffdec38aacb8f43660a5f654
from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsRegressor
from data_treatment import *





k_range = list(range(1,31)) + [50,100]
weight_options = ["uniform", "distance"]

param_grid = dict(n_neighbors = k_range, weights = weight_options)
#print (param_grid)
knn = KNeighborsRegressor()

grid = GridSearchCV(knn, param_grid, cv = 5, scoring = M_squared_error)
grid.fit(tbl,Y1)



print (grid.best_score_)
print (grid.best_params_)
print (grid.best_estimator_)

'''
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
clf = KNeighborsRegressor(algorithm='auto', leaf_size=30, metric='minkowski',
                    metric_params=None, n_jobs=None, n_neighbors=12, p=2,
                    weights='distance')
clf.fit(tbl, np.array(Y1.astype(int).values))
pred = clf.predict(tbl)
rmse(pred,Y1.values.transpose())'''

