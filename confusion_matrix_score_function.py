import os
import warnings
import numpy as np
import pandas as pd

def confusion_matrix_scoring(predicted_classes, true_classes):
    conf_matrix=np.array([
        [0,5,1,7,3,2,10],
        [30,0,20,1,6,10,1],
        [3,4,0,6,2,1,8],
        [40,5,30,0,10,20,1],
        [12,2,8,3,0,1,4],
        [6,2,3,4,1,0,6],
        [50,3,40,1,20,30,0]
    ])
    data_conf=np.array([
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0]
    ])
    for i in range(len(predicted_classes)):
        data_conf[int(predicted_classes[i])-1][int(true_classes[i])-1]+=1
    score=0
    for i in range(7):
        for j in range(7):
            score+=(data_conf[i][j]*conf_matrix[i][j])
    return score
