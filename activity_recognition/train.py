import numpy as np
from sklearn import cross_validation
from sklearn import metrics
from sklearn.metrics import roc_curve, auc

def report_result(clf, X_test, y_test, y_train):
    print("The Train count : ", len(y_train))
    print("The Test count : ", len(y_test))
    print("Accuracy : ", metrics.accuracy_score(y_test, clf.predict(X_test)))
    print(metrics.classification_report(y_test, clf.predict(X_test)))
    print(metrics.confusion_matrix(y_test, clf.predict(X_test)))
    print("\n\n")
    

def k_fold_result(clf, X, y):
    print("This is K-Fold Result ... ")
    kf = cross_validation.KFold(len(y), shuffle=True)
    for train, test in kf:
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        clf = clf.fit(X_train, y_train)
        report_result(clf, X_test, y_test, y_train)

def stratifiedk_fold_result(clf, X, y):
    print("This is StratifiedK_Fold Result ... ")
    kf = cross_validation.StratifiedKFold(y, shuffle=True)
    for train, test in kf:
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        clf = clf.fit(X_train, y_train)
        report_result(clf, X_test, y_test, y_train)

def train_test_split_result(clf, X, y):
    print("This is Random and Percentaged Spilt Result ... ")
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y)
    clf = clf.fit(X_train, y_train)
    report_result(clf, X_test, y_test, y_train)

def designated_train_test_result(train, test, clf_name):
    X_train, y_train = read_file(train_file)
    X_test, y_test = read_file(test_file)
    clf = W_Fun(clf_name)
    clf = clf.fit(X_train, y_train)
    print(metrics.classification_report(y_test, clf.predict(X_test)))
    print(metrics.confusion_matrix(y_test, clf.predict(X_test)))

def get_preds(attributes, targets, model):
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        attributes, targets, test_size=0.2)
    model.fit(X_train, y_train)
    y_true = y_test
    y_pred = model.predict(X_test)
    return (y_true, y_pred)






