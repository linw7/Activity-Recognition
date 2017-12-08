import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

#plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

if __name__ == "__main__":
    column_names = ['ID', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis']
    df = pd.read_csv("Acceleration_RAW.txt", header=None, names=column_names)
    # n = 10
    #print df.head(n)
    subject = pd.DataFrame(df["ID"].value_counts(), columns=["Count"])
    subject.index.names = ['Subject']
    #print subject.head(n)
    activities = pd.DataFrame(df["activity"].value_counts(), columns=["Count"])
    activities.index.names = ['Activity']
    #print activities.head(n)
    activity_of_subjects = pd.DataFrame(df.groupby("ID")["activity"].value_counts())
    #print activity_of_subjects.unstack().head(n)
    activity_of_subjects.unstack().plot(kind='bar', stacked=True, colormap='rainbow', title="Data Distribution Statistics")
    plt.show()