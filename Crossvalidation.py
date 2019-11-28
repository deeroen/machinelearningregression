from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from data_treatment import *
from sklearn.metrics import mean_squared_error, make_scorer
from matplotlib import pyplot as plt


def f_importances(coef, names):
    imp = coef
    imp,names = zip(*sorted(zip(imp,names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()

M_squared_error = make_scorer(mean_squared_error)
parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
svc = svm.SVR()
clf = GridSearchCV(svc, parameters, cv=3,scoring = M_squared_error)
clf.fit(tbl, np.array(Y1.astype(int).values))

sorted(clf.cv_results_.keys())

features_names = ['input1', 'input2']
f_importances(clf.best_estimator_.coef_, features_names)