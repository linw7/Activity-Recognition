import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


def read_file(file):
    data = np.loadtxt(file, delimiter=",")
    return data[:, 1:33], data[:, 0]


def get_default_clf(clf):
    if clf == "DT":
        clf = DecisionTreeClassifier()
    if clf == "RF":
        clf = RandomForestClassifier()
    if clf == "GDBT":
        clf = GradientBoostingClassifier()
    return clf
