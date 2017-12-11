from activity_recognition import utils
from activity_recognition import preprocess
from activity_recognition import calculate
from activity_recognition import train
from activity_recognition import parameter
from activity_recognition import visualization


def do_it():
    clf = utils.get_default_clf("DT")
    X, y = utils.read_file("./feature/all.csv")
    # train.k_fold_result(clf, X, y)
    parameter.random_search_parameter(clf, X, y)

do_it()
