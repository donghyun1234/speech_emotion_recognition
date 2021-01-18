import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# import ipdb

import numpy as np
import pickle
import os
from sklearn.metrics import recall_score as recall
from sklearn.metrics import confusion_matrix as confusion

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

f = open('./my_model/result/param10.pkl', 'rb')
data = pickle.load(f)
best_valid_uw, best_valid_w, pred_test_w, test_acc_w, confusion_w, pred_test_uw, test_acc_uw, confusion_uw = data

print("*****************************************************************")
print("Best valid_UA: %3.4g" % best_valid_uw)
print("Best valid_WA: %3.4g" % best_valid_w)
print('Valid Confusion Matrix:["ang","sad","hap","neu"]')
print("Test_UA: %3.4g" % test_acc_uw)
print("Test_WA: %3.4g" % test_acc_w)
print('Test Confusion Matrix:["ang","sad","hap","neu"]')
print(confusion_w)
print(confusion_uw)
print("*****************************************************************")


emotion = ["ang","sad","hap","neu"]
df_cm = pd.DataFrame(confusion_w,
        index=[i for i in emotion],
        columns = [i for i in emotion])
plt.title('confusion matrix wighted')
sns.heatmap(df_cm, annot=True)
plt.show()


df_cm = pd.DataFrame(confusion_uw,
        index=[i for i in emotion],
        columns = [i for i in emotion])

plt.title('confusion matrix unwighted')
sns.heatmap(df_cm, annot=True)
plt.show()