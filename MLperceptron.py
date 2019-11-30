from sklearn.model_selection import GridSearchCV

from sklearn.neural_network import MLPRegressor
from data_treatment import *

#Parameters

param_grid = dict(learning_rate =['constant', 'invscaling', 'adaptive'])
#print (param_grid)
LR = MLPRegressor()

grid = GridSearchCV(LR, param_grid, cv = 2, scoring = M_squared_error)
grid.fit(tbl,Y1)


print (grid.best_score_)
print (grid.best_params_)
print (grid.best_estimator_)


'''def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
clf = MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
             beta_2=0.999, early_stopping=False, epsilon=1e-08,
             hidden_layer_sizes=(100,), learning_rate='adaptive',
             learning_rate_init=0.001, max_iter=200, momentum=0.9,
             n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5,
             random_state=None, shuffle=True, solver='adam', tol=0.0001,
             validation_fraction=0.1, verbose=False, warm_start=False)
clf.fit(tbl, Y1)
pred = clf.predict(tbl)
rmse(pred,Y1.values)'''