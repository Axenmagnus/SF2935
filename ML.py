#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 11:20:24 2022

@author: magnusaxen
"""
#Preprocessing and importing of the code into python.
import pandas as pd
train_data=pd.read_csv("project_train.csv")


#Pandas subplots=True will arange the axes in a single column.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



#train_data.hist(bins=12, figsize=(15, 10))
#train_data["energy"]=(train_data['energy'] <= 0)|(train_data['energy']>=1)
#train_data["loudness"]=(train_data['loudness'] <= -100)|(train_data['loudness']>=0)
#The listed intervalls below are given in the assginment 
train_data = train_data.drop(train_data[(train_data.energy >= 1) ].index)
train_data = train_data.drop(train_data[(train_data.energy <= 0) ].index)

train_data = train_data.drop(train_data[(train_data.loudness >= 0) ].index)
train_data = train_data.drop(train_data[(train_data.loudness <= -100) ].index)

#Divided into liked label and disliked labeled, plotted to give an overview.

data_1=train_data.loc[train_data['Label'] == 1]
data_0=train_data.loc[train_data['Label'] == 0]

#data_0.hist(bins=30)#, figsize=(15, 10))
#data_1.hist(bins=30)#, figsize=(15, 10))
# %%


y_labels=train_data["Label"]
ohe = pd.get_dummies(data=train_data, columns=['key'])
ohe=ohe.drop(columns=['Label'])

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

train, test = train_test_split(train_data, test_size=0.2)
y_label_test=test["Label"]
test=test.drop(columns=["Label"])
ohe_test = pd.get_dummies(data=test, columns=['key'])

y_labels_train=train["Label"]

ohe_train = pd.get_dummies(data=train, columns=['key'])
ohe_train=ohe_train.drop(columns=['Label'])


clf = LogisticRegression(random_state=0,max_iter=10000).fit(ohe_train, y_labels_train)

clf.predict(ohe_test)
clf.predict_proba(ohe_test)
clf.score(ohe_test, y_label_test)
print(clf.score(ohe_test, y_label_test))




