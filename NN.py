#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 11:19:58 2022

@author: magnusaxen
"""
from keras import models, layers, utils, backend as K
import matplotlib.pyplot as plt
#import shap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#train_data.hist(bins=12, figsize=(15, 10))
#train_data["energy"]=(train_data['energy'] <= 0)|(train_data['energy']>=1)
#train_data["loudness"]=(train_data['loudness'] <= -100)|(train_data['loudness']>=0)
#The listed intervalls below are given in the assginment 


train_data=pd.read_csv("data1.csv")
train_data = train_data.drop(train_data[(train_data.energy >= 1) ].index)
train_data = train_data.drop(train_data[(train_data.energy <= 0) ].index)

train_data = train_data.drop(train_data[(train_data.loudness >= 0) ].index)
train_data = train_data.drop(train_data[(train_data.loudness <= -100) ].index)

#Divided into liked label and disliked labeled, plotted to give an overview.

data_1=train_data.loc[train_data['Label'] == 1]
data_0=train_data.loc[train_data['Label'] == 0]

#data_0.hist(bins=30)#, figsize=(15, 10))
#data_1.hist(bins=30)#, figsize=(15, 10))
Y=train_data["Label"]
#X=train_data.drop(columns='key')
X = pd.get_dummies(data=train_data, columns=['key'])
X=X.drop(columns=['Label'])
#X=X.drop(columns='Label')
# %% Done with data-Prepp

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(10,input_dim=len(X.columns),activation="relu"))
model.add(Dense(8,activation="relu"))
model.add(Dense(4,activation="relu"))
model.add(Dense(1,activation="sigmoid"))

model.summary()
# %%

model.compile(loss="binary_crossentropy",optimizer="rmsprop",metrics=["accuracy"])

model.fit(x=X,y=Y,epochs=200,verbose=1)





