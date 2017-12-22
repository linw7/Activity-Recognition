import ast
import math
import numpy as np 

import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE

def Change_Label(label):
    if label == "Sitting":
    	return 1
    if label == "Standing":
    	return 2
    if label == "Upstairs":
    	return 3
    if label == "Downstairs":
    	return 4
    if label == "Walking":
    	return 5
    if label == "Jogging":
    	return 6

def kurt(arr, mean, std, Seg):
    mean = float(mean)
    std = float(std)
    Seg = int(Seg)
    sum = 0
    for i in arr:
        sum = sum + math.pow((i - mean), 4)
    frac = (Seg - 1) * math.pow(std, 4) + 1
    return sum / frac

def skew(arr, mean, std, Seg):
    mean = float(mean)
    std = float(std)
    Seg = int(Seg)
    sum = 0
    for i in arr:
        sum = sum + math.pow((i - mean), 3)
    frac = (Seg - 1) * math.pow(std, 3) + 1
    return sum / frac

def rms(arr, Seg):
 	Seg = int(Seg)
 	sum = 0
 	for i in arr:
 		sum = sum + math.pow(i, 2)
 	return math.sqrt(sum / Seg)

def max(statistics_data, tag):
    max = ast.literal_eval(statistics_data[0][tag])
    for data in statistics_data:
        temp = ast.literal_eval(data[tag])
        if(temp > max):
            max = temp
    return max

def min(statistics_data, tag):
    min = ast.literal_eval(statistics_data[0][tag])
    for data in statistics_data:
        temp = ast.literal_eval(data[tag])
        if(temp < min):
            min = temp
    return min

def scale(value, max, min):
    diff = max - ast.literal_eval(value)
    range = max - min
    return int((1 - diff / range) * 255)

def Calculate_Gravity(file_dir, list):
    print("Now, Calculate Gravity ... ")
    acc_data = []
    with open(file_dir) as f:
        index = 0
        for line in f:
            clear_line = line.strip().lstrip().rstrip(';')
            raw_list = clear_line.split(',') 
            index = index + 1
            if len(raw_list) < 5:
                continue
            id = int(raw_list[0])
            status  = raw_list[1] 
            acc_x = float(raw_list[3])
            acc_y = float(raw_list[4])
            acc_z = float(raw_list[5])
            if id in list:
                gravity = math.sqrt(math.pow(acc_x, 2) + math.pow(acc_y, 2) + math.pow(acc_z, 2))
                acc_tuple = {"gravity": gravity, "acc_x":acc_x, "acc_y":acc_y, "acc_z":acc_z, "status": status}
                acc_data.append(acc_tuple)
    print("Acc_Data Length: ", len(acc_data))
    return acc_data

def Low_Pass(acc_data, k):
    print("Now, Low_Pass Filter ... ")
    index_range = len(acc_data)
    for index in range(1, index_range):
        x = acc_data[index]["acc_x"] * k + acc_data[index - 1]["acc_x"] * (1 - k)
        y = acc_data[index]["acc_y"] * k + acc_data[index - 1]["acc_y"] * (1 - k)
        z = acc_data[index]["acc_z"] * k + acc_data[index - 1]["acc_z"] * (1 - k)
        acc_data[index]["acc_x"] = acc_data[index]["acc_x"] - x
        acc_data[index]["acc_y"] = acc_data[index]["acc_y"] - y
        acc_data[index]["acc_z"] = acc_data[index]["acc_z"] - z
    print("After Low_Pass Filter, Acc_Data Length: ", len(acc_data))
    return acc_data

def Split_Data(acc_data, Seg_granularity):
    print("Now, Split Data ... ")
    splited_data = []
    gravity_cluster = []
    acc_x_cluster = []
    acc_y_cluster = []
    acc_z_cluster = []
    counter = 0
    last_status = acc_data[0]["status"]
    for acc_tuple in acc_data:
        if not (counter < Seg_granularity and acc_tuple["status"] == last_status):
            seg_data = {"status": last_status, "gravity_cluster": gravity_cluster, "acc_x_cluster": acc_x_cluster, "acc_y_cluster": acc_y_cluster, "acc_z_cluster": acc_z_cluster}
            splited_data.append(seg_data)
            gravity_cluster = []
            acc_x_cluster = []
            acc_y_cluster = []
            acc_z_cluster = []
            counter = 0
        gravity_cluster.append(acc_tuple["gravity"])
        acc_x_cluster.append(acc_tuple["acc_x"])
        acc_y_cluster.append(acc_tuple["acc_y"])
        acc_z_cluster.append(acc_tuple["acc_z"])
        last_status = acc_tuple["status"]
        counter += 1
    print("Splited_Data Length: ", len(splited_data))
    return splited_data


def Calculate_Statistic_Feature(splited_data, Seg_granularity):
    print("Now, Calculate Statistic Feature ... ")
    statistics_data = []
    for seg_data in splited_data:
        gravity_values = np.array(seg_data.pop("gravity_cluster"))
        acc_x_values = np.array(seg_data.pop("acc_x_cluster"))
        acc_y_values = np.array(seg_data.pop("acc_y_cluster"))
        acc_z_values = np.array(seg_data.pop("acc_z_cluster"))

        t_g_mean = ("%.2f" % np.mean(gravity_values))
        t_g_var = ("%.2f" % np.var(gravity_values))
        t_g_std = ("%.2f" % np.std(gravity_values))
        seg_data["g_mean"] = t_g_mean
        seg_data["g_var"] = t_g_var
        seg_data["g_std"] = t_g_std
        seg_data["g_max"] = ("%.2f" % np.max(gravity_values))
        seg_data["g_min"] = ("%.2f" % np.min(gravity_values))
        seg_data["g_kurt"] = ("%.2f" % kurt(gravity_values, t_g_mean, t_g_std, Seg_granularity))
        seg_data["g_skew"] = ("%.2f" % skew(gravity_values, t_g_mean, t_g_std, Seg_granularity))
        seg_data["g_rms"] = ("%.2f" % rms(gravity_values, Seg_granularity))

        t_x_mean = ("%.2f" % np.mean(acc_x_values))
        t_x_var = ("%.2f" % np.var(acc_x_values))
        t_x_std = ("%.2f" % np.std(acc_x_values))
        seg_data["x_mean"] = t_x_mean
        seg_data["x_var"] = t_x_var
        seg_data["x_std"] = t_x_std
        seg_data["x_max"] = ("%.2f" % np.max(acc_x_values))
        seg_data["x_min"] = ("%.2f" % np.min(acc_x_values))
        seg_data["x_kurt"] = ("%.2f" % kurt(acc_x_values, t_x_mean, t_x_std, Seg_granularity))
        seg_data["x_skew"] = ("%.2f" % skew(acc_x_values, t_x_mean, t_x_std, Seg_granularity))
        seg_data["x_rms"] = ("%.2f" % rms(acc_x_values, Seg_granularity))

        t_y_mean = ("%.2f" % np.mean(acc_y_values))
        t_y_var = ("%.2f" % np.var(acc_y_values))
        t_y_std = ("%.2f" % np.std(acc_y_values))
        seg_data["y_mean"] = t_y_mean
        seg_data["y_var"] = t_y_var
        seg_data["y_std"] = t_y_std
        seg_data["y_max"] = ("%.2f" % np.max(acc_y_values))
        seg_data["y_min"] = ("%.2f" % np.min(acc_y_values))
        seg_data["y_kurt"] = ("%.2f" % kurt(acc_y_values, t_y_mean, t_y_std, Seg_granularity))
        seg_data["y_skew"] = ("%.2f" % skew(acc_y_values, t_y_mean, t_y_std, Seg_granularity))
        seg_data["y_rms"] = ("%.2f" % rms(acc_y_values, Seg_granularity))

        t_z_mean = ("%.2f" % np.mean(acc_z_values))
        t_z_var = ("%.2f" % np.var(acc_z_values))
        t_z_std = ("%.2f" % np.std(acc_z_values))
        seg_data["z_mean"] = t_z_mean
        seg_data["z_var"] = t_z_var
        seg_data["z_std"] = t_z_std
        seg_data["z_max"] = ("%.2f" % np.max(acc_z_values))
        seg_data["z_min"] = ("%.2f" % np.min(acc_z_values))
        seg_data["z_kurt"] = ("%.2f" % kurt(acc_z_values, t_z_mean, t_z_std, Seg_granularity))
        seg_data["z_skew"] = ("%.2f" % skew(acc_z_values, t_z_mean, t_z_std, Seg_granularity))
        seg_data["z_rms"] = ("%.2f" % rms(acc_z_values, Seg_granularity))
        statistics_data.append(seg_data)
    print("Statistics_Data Length", len(statistics_data))
    return statistics_data

def Scale_Feature(statistics_data):
    print("Now, Scale ... ")
    g_mean_min = min(statistics_data, "g_mean")
    g_mean_max = max(statistics_data, "g_mean")
    g_var_min = min(statistics_data, "g_var")
    g_var_max = max(statistics_data, "g_var")
    g_std_min = min(statistics_data, "g_std")
    g_std_max = max(statistics_data, "g_std")
    g_max_min = min(statistics_data, "g_max")
    g_max_max = max(statistics_data, "g_max")
    g_min_min = min(statistics_data, "g_min")
    g_min_max = max(statistics_data, "g_min")
    g_kurt_min = min(statistics_data, "g_kurt")
    g_kurt_max = max(statistics_data, "g_kurt")
    g_skew_min = min(statistics_data, "g_skew")
    g_skew_max = max(statistics_data, "g_skew")
    g_rms_min = min(statistics_data, "g_rms")
    g_rms_max = max(statistics_data, "g_rms")

    x_mean_min = min(statistics_data, "x_mean")
    x_mean_max = max(statistics_data, "x_mean")
    x_var_min = min(statistics_data, "x_var")
    x_var_max = max(statistics_data, "x_var")
    x_std_min = min(statistics_data, "x_std")
    x_std_max = max(statistics_data, "x_std")
    x_max_min = min(statistics_data, "x_max")
    x_max_max = max(statistics_data, "x_max")
    x_min_min = min(statistics_data, "x_min")
    x_min_max = max(statistics_data, "x_min")
    x_kurt_min = min(statistics_data, "x_kurt")
    x_kurt_max = max(statistics_data, "x_kurt")
    x_skew_min = min(statistics_data, "x_skew")
    x_skew_max = max(statistics_data, "x_skew")
    x_rms_min = min(statistics_data, "x_rms")
    x_rms_max = max(statistics_data, "x_rms")

    y_mean_min = min(statistics_data, "y_mean")
    y_mean_max = max(statistics_data, "y_mean")
    y_var_min = min(statistics_data, "y_var")
    y_var_max = max(statistics_data, "y_var")
    y_std_min = min(statistics_data, "y_std")
    y_std_max = max(statistics_data, "y_std")
    y_max_min = min(statistics_data, "y_max")
    y_max_max = max(statistics_data, "y_max")
    y_min_min = min(statistics_data, "y_min")
    y_min_max = max(statistics_data, "y_min")
    y_kurt_min = min(statistics_data, "y_kurt")
    y_kurt_max = max(statistics_data, "y_kurt")
    y_skew_min = min(statistics_data, "y_skew")
    y_skew_max = max(statistics_data, "y_skew")
    y_rms_min = min(statistics_data, "y_rms")
    y_rms_max = max(statistics_data, "y_rms")

    z_mean_min = min(statistics_data, "z_mean")
    z_mean_max = max(statistics_data, "z_mean")
    z_var_min = min(statistics_data, "z_var")
    z_var_max = max(statistics_data, "z_var")
    z_std_min = min(statistics_data, "z_std")
    z_std_max = max(statistics_data, "z_std")
    z_max_min = min(statistics_data, "z_max")
    z_max_max = max(statistics_data, "z_max")
    z_min_min = min(statistics_data, "z_min")
    z_min_max = max(statistics_data, "z_min")
    z_kurt_min = min(statistics_data, "z_kurt")
    z_kurt_max = max(statistics_data, "z_kurt")
    z_skew_min = min(statistics_data, "z_skew")
    z_skew_max = max(statistics_data, "z_skew")
    z_rms_min = min(statistics_data, "z_rms")
    z_rms_max = max(statistics_data, "z_rms")

    scaling_data = []
    for scale_data in statistics_data:
        scale_data["g_mean"] = scale(scale_data["g_mean"], g_mean_max, g_mean_min)
        scale_data["g_var"] = scale(scale_data["g_var"], g_var_max, g_var_min)
        scale_data["g_std"] = scale(scale_data["g_std"], g_std_max, g_std_min)
        scale_data["g_max"] = scale(scale_data["g_max"], g_max_max, g_max_min)
        scale_data["g_min"] = scale(scale_data["g_min"], g_min_max, g_min_min)
        scale_data["g_kurt"] = scale(scale_data["g_kurt"], g_kurt_max, g_kurt_min)
        scale_data["g_skew"] = scale(scale_data["g_skew"], g_skew_max, g_skew_min)
        scale_data["g_rms"] = scale(scale_data["g_rms"], g_rms_max, g_rms_min)

        scale_data["x_mean"] = scale(scale_data["x_mean"], x_mean_max, x_mean_min)
        scale_data["x_var"] = scale(scale_data["x_var"], x_var_max, x_var_min)
        scale_data["x_std"] = scale(scale_data["x_std"], x_std_max, x_std_min)
        scale_data["x_max"] = scale(scale_data["x_max"], x_max_max, x_max_min)
        scale_data["x_min"] = scale(scale_data["x_min"], x_min_max, x_min_min)
        scale_data["x_kurt"] = scale(scale_data["x_kurt"], x_kurt_max, x_kurt_min)
        scale_data["x_skew"] = scale(scale_data["x_skew"], x_skew_max, x_skew_min)
        scale_data["x_rms"] = scale(scale_data["x_rms"], x_rms_max, x_rms_min)

        scale_data["y_mean"] = scale(scale_data["y_mean"], y_mean_max, y_mean_min)
        scale_data["y_var"] = scale(scale_data["y_var"], y_var_max, y_var_min)
        scale_data["y_std"] = scale(scale_data["y_std"], y_std_max, y_std_min)
        scale_data["y_max"] = scale(scale_data["y_max"], y_max_max, y_max_min)
        scale_data["y_min"] = scale(scale_data["y_min"], y_min_max, y_min_min)
        scale_data["y_kurt"] = scale(scale_data["y_kurt"], y_kurt_max, y_kurt_min)
        scale_data["y_skew"] = scale(scale_data["y_skew"], y_skew_max, y_skew_min)
        scale_data["y_rms"] = scale(scale_data["y_rms"], y_rms_max, y_rms_min)

        scale_data["z_mean"] = scale(scale_data["z_mean"], z_mean_max, z_mean_min)
        scale_data["z_var"] = scale(scale_data["z_var"], z_var_max, z_var_min)
        scale_data["z_std"] = scale(scale_data["z_std"], z_std_max, z_std_min)
        scale_data["z_max"] = scale(scale_data["z_max"], z_max_max, z_max_min)
        scale_data["z_min"] = scale(scale_data["z_min"], z_min_max, z_min_min)
        scale_data["z_kurt"] = scale(scale_data["z_kurt"], z_kurt_max, z_kurt_min)
        scale_data["z_skew"] = scale(scale_data["z_skew"], z_skew_max, z_skew_min)
        scale_data["z_rms"] = scale(scale_data["z_rms"], z_rms_max, z_rms_min)
    print("Done, Calculate ! ")
    return statistics_data


c_sit = 0
c_stand = 0
c_upstairs = 0
c_downstairs = 0
c_walk = 0
c_jog = 0


def count_label(label):
    global c_sit
    global c_stand
    global c_upstairs
    global c_downstairs
    global c_walk
    global c_jog

    if label == "1":
        c_sit = c_sit + 1
    if label == "2":
        c_stand = c_stand + 1
    if label == "3":
        c_upstairs = c_upstairs + 1
    if label == "4":
        c_downstairs = c_downstairs + 1
    if label == "5":
        c_walk = c_walk + 1
    if label == "6":
        c_jog = c_jog + 1



def count_print():
    global c_sit
    global c_stand
    global c_upstairs
    global c_downstairs
    global c_walk
    global c_jog
    
    print("Sit :", c_sit)
    print("Stand :", c_stand)
    print("Upstairs :", c_upstairs)
    print("Downstairs :", c_downstairs)
    print("Walk :", c_walk)
    print("Jog :", c_jog)


def Write_File(write_file, statistics_data):
    print("Now, Write the File ! ")
    file = "./feature/" + write_file
    with open(file, "w+") as the_file:
        for seg_data in statistics_data:
            label = Change_Label(str(seg_data["status"]))
            # count_label(str(label))
            # label = str(seg_data["status"])
            row = str(label) + "," + \
                str(seg_data["g_mean"]) + "," + str(seg_data["g_var"]) + "," + str(seg_data["g_std"]) + "," + str(seg_data["g_max"]) + "," + str(seg_data["g_min"]) + "," + str(seg_data["g_kurt"]) + "," + str(seg_data["g_skew"]) + "," + str(seg_data["g_rms"]) + "," + \
                str(seg_data["x_mean"]) + "," + str(seg_data["x_var"]) + "," + str(seg_data["x_std"]) + "," + str(seg_data["x_max"]) + "," + str(seg_data["x_min"]) + "," + str(seg_data["x_kurt"]) + "," + str(seg_data["x_skew"]) + "," + str(seg_data["x_rms"]) + "," + \
                str(seg_data["y_mean"]) + "," + str(seg_data["y_var"]) + "," + str(seg_data["y_std"]) + "," + str(seg_data["y_max"]) + "," + str(seg_data["y_min"]) + "," + str(seg_data["y_kurt"]) + "," + str(seg_data["y_skew"]) + "," + str(seg_data["y_rms"]) + "," + \
                str(seg_data["z_mean"]) + "," + str(seg_data["z_var"]) + "," + str(seg_data["z_std"]) + "," + str(seg_data["z_max"]) + "," + str(seg_data["z_min"]) + "," + str(seg_data["z_kurt"]) + "," + str(seg_data["z_skew"]) + "," + str(seg_data["z_rms"]) + "\n"
            the_file.write(row)
    count_print()

def write_resamble(X_resampled, y_resampled, read_file):
    file = "./feature/" + "smoted_"  + read_file
    n_resample = len(y_resampled)
    data = pd.DataFrame(X_resampled)
    with open(file, "w+") as the_file:
        for i in range(0, n_resample):
            count_label(str(y_resampled[i]))
            row = str(y_resampled[i]) + "," + \
                str(int(data.ix[i, 0])) + "," + str(int(data.ix[i, 1])) + "," + str(int(data.ix[i, 2])) + "," + str(int(data.ix[i, 3])) + "," + \
                str(int(data.ix[i, 4])) + "," + str(int(data.ix[i, 5])) + "," + str(int(data.ix[i, 6])) + "," + str(int(data.ix[i, 7])) + "," + \
                str(int(data.ix[i, 8])) + "," + str(int(data.ix[i, 9])) + "," + str(int(data.ix[i, 10])) + "," + str(int(data.ix[i, 11])) + "," + \
                str(int(data.ix[i, 12])) + "," + str(int(data.ix[i, 13])) + "," + str(int(data.ix[i, 14])) + "," + str(int(data.ix[i, 15])) + "," + \
                str(int(data.ix[i, 16])) + "," + str(int(data.ix[i, 17])) + "," + str(int(data.ix[i, 18])) + "," + str(int(data.ix[i, 19])) + "," + \
                str(int(data.ix[i, 20])) + "," + str(int(data.ix[i, 21])) + "," + str(int(data.ix[i, 22])) + "," + str(int(data.ix[i, 23])) + "," + \
                str(int(data.ix[i, 24])) + "," + str(int(data.ix[i, 25])) + "," + str(int(data.ix[i, 26])) + "," + str(int(data.ix[i, 27])) + "," + \
                str(int(data.ix[i, 28])) + "," + str(int(data.ix[i, 29])) + "," + \
                str(int(data.ix[i, 30])) + "," + str(int(data.ix[i, 31])) + "\n"
            the_file.write(row)
    count_print()


def smote(read_file):
    print("Now, Start Smote ... ")
    file = "./feature/" + read_file
    data = pd.read_csv(file)
    array = data.values
    X = array[:, 1:33]
    y = array[:, 0]
    sm = SMOTE(k_neighbors=5, kind='borderline1')
    X_resampled, y_resampled = sm.fit_sample(X, y)
    write_resamble(X_resampled, y_resampled, read_file)
    print("Done !")


def preprocess(file_dir, write_file, list, Seg_granularity, Low_Pass_k):
    print("The Id List : ", list)
    acc_data = Calculate_Gravity(file_dir, list)
    # acc_data = Low_Pass(acc_data, Low_Pass_k)
    splited_data = Split_Data(acc_data, Seg_granularity)
    statistics_data = Calculate_Statistic_Feature(splited_data, Seg_granularity)
    statistics_data = Scale_Feature(statistics_data)
    Write_File(write_file, statistics_data)
    # smote(write_file)

# Do it !    
def calculate_range_id(start, end):
    list = []
    for i in range(start, end + 1):
        list.append(i)
    preprocess("./data_set/acceleration_raw.txt", str(start) +
               "-" + str(end) + ".txt", list, 40, 0.1)

def calculate_except_range_id(start, end):
    list = []
    for i in range(1, 37):
        list.append(i)
    for j in range(start, end + 1):
        list.remove(j)
    preprocess("./data_set/acceleration_raw.txt", "except_" +
               str(start) + "-" + str(end) + ".csv", list, 40, 0.1)

def calculate_id(id):
    list = []
    list.append(id)
    preprocess("./data_set/acceleration_raw.txt",
               str(id) + ".csv", list, 40, 0.1)

def calculate_per_id():
    for i in range(1, 37):
        calculate_id(i)

def calculate_expect_id(id):
    list = []
    for i in range(1, 37):
        list.append(i)
    list.remove(id)
    preprocess("./data_set/acceleration_raw.txt", "except_" +
               str(id) + ".csv", list, 40, 0.1)

def calculate_per_expect_id():
    for i in range(1, 37):
        calculate_expect_id(i)

def calculate_all():
    list = []
    for i in range(1, 37):
        list.append(i)
    preprocess("./data_set/acceleration_raw.txt",
               "all" + ".csv", list, 40, 0.1)




























