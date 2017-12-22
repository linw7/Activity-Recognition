from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import ListedColormap
import numpy as np

import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname='/home/tk/Desktop/msyh.ttf')


def plot_learning_curve_default(X, y, clf):
    plt.title("Learning Curve")
    plt.xlabel("Training Instance")
    plt.ylabel("Score")

    part = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

    train_sizes, train_scores, test_scores = learning_curve(clf, X, y, cv=part, n_jobs=1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score")
    plt.legend(loc="best")

    plt.show()
    plt.close()


def plot_learning_curve_cv(X, y, clf, cv):


    plt.title("Learning Curve")
    plt.xlabel("Training Instance")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(clf, X, y, cv=cv, n_jobs=1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, "o-",
             color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, "o-",
             color="g", label="Cross-validation score")
    plt.legend(loc="best")

    plt.show()
    plt.close()


def plot_paramter_curve_default(X, y, clf, param_name, param_range):
    plt.title('Validation Curve')
    plt.xlabel('Max_Depth')
    plt.ylabel('Score')

    part = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

    train_scores, test_scores = validation_curve(clf, X, y, param_name=param_name, param_range=param_range, cv=part, n_jobs=1)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(p_range, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(p_range, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(p_range, train_scores_mean, "o-",
             color="r", label='Training Score')
    plt.plot(p_range, test_scores_mean, "o-",
             color="g", label='Cross-Validation Score')
    plt.legend(loc='best')

    plt.show()
    plt.close()


def plot_paramter_curve_cv(X, y, clf, cv, param_name, param_range):
    plt.title('Validation Curve')
    plt.xlabel('Max_Depth')
    plt.ylabel('Score')

    train_scores, test_scores = validation_curve(
        clf, X, y, param_name=param_name, param_range=p_range, cv=cv, n_jobs=1)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(p_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(p_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(p_range, train_scores_mean, "o-",
             color="r", label='Training Score')
    plt.plot(p_range, test_scores_mean, "o-",
             color="g", label='Cross-Validation Score')
    plt.legend(loc='best')

    plt.show()
    plt.close()


def plot_confusion_matrix(confusion_matrix):
    plt.title("Confusion Matrix")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    LABELS = ["Sitting", "Standing", "Upstairs", "Doenstairs", "Walking", "Jogging"]

    normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32) / np.sum(confusion_matrix) * 100
    
    plt.imshow(normalised_confusion_matrix, interpolation='nearest', cmap=plt.cm.rainbow)
    plt.colorbar()

    tick_marks = np.arange(6)
    plt.xticks(tick_marks, LABELS, rotation=90)
    plt.yticks(tick_marks, LABELS)
 
    plt.show()
    plt.close()


def plot_gridsearch(clf, X, y):
    plt.title("Parameters Matrix")
    plt.xlabel('Max_Depth')
    plt.ylabel('Min_Samples_Leaf')

    depth_range = np.linspace(1, 5, 5)
    min_leaf_range = np.linspace(0.01, 0.05, 5)

    param = dict(max_depth=depth_range, min_samples_leaf=min_leaf_range)
    grid_search = GridSearchCV(clf, param_grid=param, cv=2, scoring='accuracy')
    grid_search.fit(X, y)

    scores = [x[1] for x in grid_search.grid_scores_]
    scores = np.array(scores).reshape(len(depth_range), len(min_leaf_range))

    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.rainbow)
    plt.colorbar()
    
    plt.xticks(np.arange(len(depth_range)), depth_range, rotation=45)
    plt.yticks(np.arange(len(min_leaf_range)), min_leaf_range)
   
    plt.show()
    plt.close()


def plot_learning_curve_cv_compare(X, y, X_S, y_S, clf, cv):

    plt.subplot(121)
    plt.title("未均衡处理前随机森林学习曲线", fontproperties=myfont, fontsize=12)
    plt.xlabel("训练样例数", fontproperties=myfont, fontsize=12)
    plt.ylabel("模型精度", fontproperties=myfont, fontsize=12)

    train_sizes, train_scores, test_scores = learning_curve(
        clf, X, y, cv=cv, n_jobs=1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, "o-",
             color="r", label="训练集精度")
    plt.plot(train_sizes, test_scores_mean, "o-",
             color="g", label="十折交叉精度")
    plt.legend(loc="best", prop=myfont, fontsize=12)


    plt.subplot(122)
    plt.title("均衡处理后随机森林学习曲线", fontproperties=myfont, fontsize=12)
    plt.xlabel("训练样例数", fontproperties=myfont, fontsize=12)
    plt.ylabel("模型精度", fontproperties=myfont, fontsize=12)
    train_sizes, train_scores, test_scores = learning_curve(
        clf, X_S, y_S, cv=cv, n_jobs=1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, "o-",
             color="r", label="训练集精度")
    plt.plot(train_sizes, test_scores_mean, "o-",
             color="g", label="十折交叉精度")
    plt.legend(loc="best", prop=myfont, fontsize=12)


    plt.show()
    plt.close()
