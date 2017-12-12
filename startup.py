from activity_recognition import utils
from activity_recognition import preprocess
from activity_recognition import calculate
from activity_recognition import train
from activity_recognition import parameter
from activity_recognition import visualization
from activity_recognition import selection


def do_it():
   
    #X = utils.read_feature("./feature/20/Upstairs.csv")
    #Y = utils.read_feature("./feature/21/Upstairs.csv")
    #selection.activity_pearson(X, Y)
    id_list = [3, 5]
    feature_list = [0,1,2,3,4,5,6]
    utils.extract_activuty_feature(id_list, feature_list)

do_it()
