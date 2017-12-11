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
