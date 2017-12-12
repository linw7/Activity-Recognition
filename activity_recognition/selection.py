import numpy as np
import math

def activity_pearson(X, Y):
    x = []
    y = []
    
    line_feature = X[0]
    column = len(line_feature)

    sum_pearson = 0
    for i in range(0, column):
        for data in X:
            x.append(data[i])

        for data in Y:
            y.append(data[i])

        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        len_x = len(x)
        len_y = len(y)

        if len_x != len_y:
            print("Not equal !")
            continue

        sum_son = 0
        for j in range(0, len_x):
            sum_son = sum_son + (x[j] - x_mean) * (y[j] - y_mean)

        sum_ma_1 = 0
        sum_ma_2 = 0
        for j in range(0, len_x):
            sum_ma_1 = sum_ma_1 + (x[j] - x_mean) * (x[j] - x_mean)
            sum_ma_2 = sum_ma_2 + (y[j] - y_mean) * (y[j] - y_mean)

        sum_ma = math.sqrt(sum_ma_1 * sum_ma_2)

        pearson_feature = sum_son / sum_ma
        sum_pearson = sum_pearson + pearson_feature
        print("The", i + 1, "feature pearson coefficient : ", pearson_feature)
    
    print("Activity avg_pearson : ", sum_pearson / column)
