import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

def read_file(file):
    data = np.loadtxt(file, delimiter=",")
    return data[:, 1:33], data[:, 0]

def read_feature(file):
    data = np.loadtxt(file, delimiter=",")
    return data[:, 1:33]

def get_default_clf(clf):
    if clf == "DT":
        clf = DecisionTreeClassifier()
    if clf == "RF":
        clf = RandomForestClassifier()
    if clf == "GDBT":
        clf = GradientBoostingClassifier()
    return clf

def line_data(line):
    row = str(int(line[0])) + "," + \
        str(int(line[1])) + "," + str(int(line[2])) + "," + str(int(line[3])) + "," + str(int(line[4])) + "," + str(int(line[5])) + "," + str(int(line[6])) + "," + str(int(line[7])) + "," + str(int(line[8])) + "," + \
        str(int(line[9])) + "," + str(int(line[10])) + "," + str(int(line[11])) + "," + str(int(line[12])) + "," + str(int(line[13])) + "," + str(int(line[14])) + "," + str(int(line[15])) + "," + str(int(line[16])) + "," + \
        str(int(line[17])) + "," + str(int(line[18])) + "," + str(int(line[19])) + "," + str(int(line[20])) + "," + str(int(line[21])) + "," + str(int(line[22])) + "," + str(int(line[23])) + "," + str(int(line[24])) + "," + \
        str(int(line[25])) + "," + str(int(line[26])) + "," + str(int(line[27])) + "," + str(int(line[28])) + "," + str(int(line[29])) + "," + str(int(line[30])) + "," + str(int(line[31])) + "," + str(int(line[32])) + "\n"
    return row

def write_line(line, tag, i):
    if tag == 1:
        file = "./feature/" + str(i) + "/Sitting.csv"
        with open(file, "a") as the_file:
            row = line_data(line)
            the_file.write(row)
    if tag == 2:
        file = "./feature/" + str(i) + "/Standing.csv"
        with open(file, "a") as the_file:
            row = line_data(line)
            the_file.write(row)
    if tag == 3:
        file = "./feature/" + str(i) + "/Upstairs.csv"
        with open(file, "a") as the_file:
            row = line_data(line)
            the_file.write(row)
    if tag == 4:
        file = "./feature/" + str(i) + "/Downstairs.csv"
        with open(file, "a") as the_file:
            row = line_data(line)
            the_file.write(row)
    if tag == 5:
        file = "./feature/" + str(i) + "/Walking.csv"
        with open(file, "a") as the_file:
            row = line_data(line)
            the_file.write(row)
    if tag == 6:
        file = "./feature/" + str(i) + "/Jogging.csv"
        with open(file, "a") as the_file:
            row = line_data(line)
            the_file.write(row)

def extract_activity(file, i):
    data = np.loadtxt(file, delimiter=",")
    X = data[:, 0:33]

    Sitting = 0
    Standing = 0
    Upstairs = 0
    Downstairs = 0
    Walking = 0
    Jogging = 0

    for line in X:
        if line[0] == 1:
            Sitting = Sitting + 1
            if Sitting > 5 and Sitting <= 15:
                write_line(line, line[0], i)
        if line[0] == 2:
            Standing = Standing + 1
            if Standing > 5 and Standing <= 15:
                write_line(line, line[0], i)
        if line[0] == 3:
            Upstairs = Upstairs + 1
            if Upstairs > 10 and Upstairs <= 30:
                write_line(line, line[0], i)
        if line[0] == 4:
            Downstairs = Downstairs + 1
            if Downstairs > 10 and Downstairs <= 30:
                write_line(line, line[0], i)
        if line[0] == 5:
            Walking = Walking + 1
            if Walking > 10 and Walking <= 30:
                write_line(line, line[0], i)
        if line[0] == 6:
            Jogging = Jogging + 1
            if Jogging > 10 and Jogging <= 30:
                write_line(line, line[0], i)
        
def extract_per_people():
    list = []
    for i in range(1, 37):
        file = "./feature/" + str(i) + ".csv"
        extract_activity(file, i)

def extract_feature(file, write_file, list):
    # min list item is 0 ! 
    X, y = read_file(file)
    length = len(y)
    with open(write_file, "w+") as the_file:
        for i in range(0, length):
            row = str(int(y[i]))
            for j in list:
                row = row + "," + str(int(X[i][j]))
            row = row + "\n"
            the_file.write(row)

def hash_name(list):
    list_index = []
    for name in list:
        if name == "g_mean":
            list_index.append(0)
        if name == "g_var":
            list_index,append(1)
        if name == "g_std":
            list_index.append(2)
        if name == "g_max":
            list_index.append(3)
        if name == "g_min":
            list_index.append(4)
        if name == "g_kurt":
            list_index.append(5)
        if name == "g_skew":
            list_index.append(6)
        if name == "g_rms":
            list_index.append(7)
        if name == "x_mean":
            list_index.append(8)
        if name == "x_var":
            list_index, append(9)
        if name == "x_std":
            list_index.append(10)
        if name == "x_max":
            list_index.append(11)
        if name == "x_min":
            list_index.append(12)
        if name == "x_kurt":
            list_index.append(13)
        if name == "x_skew":
            list_index.append(14)
        if name == "x_rms":
            list_index.append(15)
        if name == "y_mean":
            list_index.append(16)
        if name == "y_var":
            list_index, append(17)
        if name == "y_std":
            list_index.append(18)
        if name == "y_max":
            list_index.append(19)
        if name == "y_min":
            list_index.append(20)
        if name == "y_kurt":
            list_index.append(21)
        if name == "y_skew":
            list_index.append(22)
        if name == "y_rms":
            list_index.append(23)
        if name == "z_mean":
            list_index.append(24)
        if name == "z_var":
            list_index, append(25)
        if name == "z_std":
            list_index.append(26)
        if name == "z_max":
            list_index.append(27)
        if name == "z_min":
            list_index.append(28)
        if name == "z_kurt":
            list_index.append(29)
        if name == "z_skew":
            list_index.append(30)
        if name == "z_rms":
            list_index.append(31)
    return list_index


def extract_activuty_feature(id_list, feature_list):
    for id in id_list:
        file1 = "./feature/" + str(id) + "/Sitting.csv"
        file1_extract = "./feature/" + str(id) + "/Extract_Sitting.csv"
        extract_feature(file1, file1_extract, feature_list)

        file2 = "./feature/" + str(id) + "/Standing.csv"
        file2_extract = "./feature/" + str(id) + "/Extract_Standing.csv"
        extract_feature(file2, file2_extract, feature_list)

        file3 = "./feature/" + str(id) + "/Upstairs.csv"
        file3_extract = "./feature/" + str(id) + "/Extract_Upstairs.csv"
        extract_feature(file3, file3_extract, feature_list)

        file4 = "./feature/" + str(id) + "/Downstairs.csv"
        file4_extract = "./feature/" + str(id) + "/Extract_Downstairs.csv"
        extract_feature(file4, file4_extract, feature_list)

        file5 = "./feature/" + str(id) + "/Walking.csv"
        file5_extract = "./feature/" + str(id) + "/Extract_Walking.csv"
        extract_feature(file5, file5_extract, feature_list)

        file6 = "./feature/" + str(id) + "/Jogging.csv"
        file6_extract = "./feature/" + str(id) + "/Extract_Jogging.csv"
        extract_feature(file6, file6_extract, feature_list)
        
        
        
        
        
        

