from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from time import time
import numpy as np   

def grid_search_parameter(clf, X, y):
    param = {'max_depth':[3,4,5,6,7,8]}
    grid_search = GridSearchCV(clf, param, cv=5, scoring='accuracy')
    start = time()
    grid_search.fit(X, y)
    print("GridSearchCV took %.3f seconds for parameter settings." % (time() - start))
    print("\n", "Beat Estimator : ", grid_search.best_estimator_)
    print("\n", "Grid Score : ", grid_search.grid_scores_)
    print("\n", "Best Score : ", grid_search.best_score_)
    print("\n", "Best Param : ", grid_search.best_params_)

def random_search_parameter(clf, X, y):
    param = {"max_depth": [3, 4, 5],
              "min_samples_split": sp_randint(2, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "criterion": ['gini', 'entropy']
              }
    random_search = RandomizedSearchCV(clf, param, cv=5, scoring='accuracy')
    start = time()
    random_search.fit(X, y)
    print("RandomizedSearchCV took %.3f seconds for parameter settings." % (time() - start))
    for i in range(1, 10):
        candidates = np.flatnonzero(random_search.cv_results_['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank:{0}".format(i))
            print("Mean validation score:{0:.3f}(std:{1:.3f})".format(
                random_search.cv_results_['mean_test_score'][candidate],
                random_search.cv_results_['std_test_score'][candidate]))
            print("Parameters:{0}".format(random_search.cv_results_['params'][candidate]), "\n")
