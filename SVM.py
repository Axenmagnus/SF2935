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
from sklearn.decomposition import PCA
  # %%
  
  
# import numpy as np
# import cvxopt
# from utils import plot_contour, create_dataset


# def linear(x, z):
#     return np.dot(x, z.T)


# def polynomial(x, z, p=5):
#     return (1 + np.dot(x, z.T)) ** p


# def gaussian(x, z, sigma=0.1):
#     #global x
#     #global z
#     l=x
#     r=z
#     return np.exp(-np.linalg.norm(x - z, axis=1) ** 2 / (2 * (sigma ** 2)))


# class SVM:
#     def __init__(self, kernel=gaussian, C=1):
#         self.kernel = kernel
#         self.C = C

#     def fit(self, X, y):
#         self.y = y
#         self.X = X
#         m, n = X.shape

#         # Calculate Kernel
#         self.K = np.zeros((m, m))
#         for i in range(m):
#             self.K[i, :] = self.kernel(X[i, np.newaxis], self.X)

#         # Solve with cvxopt final QP needs to be reformulated
#         # to match the input form for cvxopt.solvers.qp
#         P = cvxopt.matrix(np.outer(y, y) * self.K)
#         q = cvxopt.matrix(-np.ones((m, 1)))
#         G = cvxopt.matrix(np.vstack((np.eye(m) * -1, np.eye(m))))
#         h = cvxopt.matrix(np.hstack((np.zeros(m), np.ones(m) * self.C)))
#         A = cvxopt.matrix(y, (1, m), "d")
#         b = cvxopt.matrix(np.zeros(1))
#         cvxopt.solvers.options["show_progress"] = False
#         sol = cvxopt.solvers.qp(P, q, G, h, A, b)
#         self.alphas = np.array(sol["x"])

#     def predict(self, X):
#         y_predict = np.zeros((X.shape[0]))
#         sv = self.get_parameters(self.alphas)
        
#         for i in range(X.shape[0]):
#             #print(X[i])
#             #print(self.X[sv])
#             #print(len(X[i]))
#             #print(len(self.X[sv]))
#             #print(i)
#             y_predict[i] = np.sum(
#                 self.alphas[sv]
#                 * self.y[sv, np.newaxis]
#                 * self.kernel(X[i], self.X[sv])[:, np.newaxis]
#             )

#         return np.sign(y_predict + self.b)

#     def get_parameters(self, alphas):
#         threshold = 1e-5

#         sv = ((alphas > threshold) * (alphas < self.C)).flatten()
#         self.w = np.dot(X[sv].T, alphas[sv] * self.y[sv, np.newaxis])
#         self.b = np.mean(
#             self.y[sv, np.newaxis]
#             - self.alphas[sv] * self.y[sv, np.newaxis] * self.K[sv, sv][:, np.newaxis]
#         )
#         return sv


# if __name__ == "__main__":
#     np.random.seed(1)
#     y=train_data[["Label"]].values
#     X=train_data.drop("Label",axis=1).values
#     Y_new=np.zeros(len(y))

#     for i in range(len(y)):
#         Y_new[i]=y[i]
#     #X, y = create_dataset(N=50)

#     svm = SVM(kernel=polynomial)
#     svm.fit(X, y)
#     y_pred = svm.predict(X)
#     plot_contour(X, y, svm)

#     print(f"Accuracy: {sum(y==y_pred)/y.shape[0]}")

  
  
  # %%
import numpy , random , math
from scipy . optimize import minimize
import matplotlib.pyplot as plt
import numpy as np


def kernel(x,y,kType):
    p=10
    sigma=2
    if kType=="linear":
        val= np.dot(x,y)
    elif kType=="polynomial":
        val=(np.dot(x,y)+1)**p
    elif kType=="RBF":
        val=math.exp(-math.pow(numpy.linalg.norm(numpy.subtract(x, y)), 2)/(2 * math.pow(sigma,2)))
    else:
        return ValueError("Nan")
    return val



def objective(alpha):
    
    return (1/2)*numpy.dot(alpha, numpy.dot(alpha, P)) - numpy.sum(alpha)





def zerofun(alpha):
    summ=np.dot(alpha,t)
    return summ
    

def b(alpha,kType):
    bsum = 0
    for value in nonzero:
        value=value[0]
        print(value[0])
        print(value[1])
        print(value[2])
        bsum += value[0] * value[2] * kernel(value[1], nonzero[0][1],kType)
    return bsum - nonzero[0][2]





def ind(alpha,x, y,kType, b):
    totsum = 0
    for value in nonzero:
        totsum += value[0] * value[2] * kernel([x, y], value[1],kType)
    return totsum - b






t=np.zeros(2)
P=np.zeros((2,2))







inputs=train_data.drop("Label",axis=1)
targets=train_data[["Label"]]
N = inputs . shape [0] 
permute=list(range(N))
random .shuffle(permute)
x= inputs.values [ permute , : ]
t=targets.values[ permute ]
start=np.zeros(N)



# Creating the P matrix
# Specify kType below

P= np.zeros((N,N))

Ktype="RBF"
for i in range(N):
    for j in range(N):
        P[i,j]=t[i]*t[j]*kernel(x[i],x[j],Ktype)


start=numpy.zeros(N)
# As for the upper constraint is it just arbitrary?
# How to specify a good one ?
C=None
XC={"type":"eq", "fun":zerofun}

B = [(0, C) for i in range(N)]
    

ret = minimize ( objective , start ,bounds=B, constraints=XC )
alpha = ret['x']
alpha = np.array([round(i, 5) for i in alpha])
        



## Plotting




plt.plot(data_0 ,data_1 ,'b .')

#plt.plot([p[0] for p in classB ] ,[ p [ 1 ] for p in classB ] ,'r.' )
plt.axis('equal') # Force same s c a l e on both axes
j=0
nonzero=[]
for i in range((N)):
    #print(i)
    #print(j)
    if alpha[i] > 10e-5:
        nonzero.append( [(alpha[i], inputs.values[i,:], targets.values[i][0])])
        j=j+1
    
    
    
#nonzero = [(alpha[i], inputs[i], targets[i]) for i in range(N)]
xgrid=numpy.linspace(-5,5)
ygrid=numpy.linspace(-4,4)
b=b(alpha,Ktype)
grid=numpy.array([[ind(alpha,x,y,Ktype,b)
for x in xgrid ]
for y in ygrid ] )

plt.contour( xgrid , ygrid , grid ,
(-1,0,1),
colors=('red' ,'black', 'blue') ,
linewidths=(1 , 3 , 1))
plt.show()# Show the p l o t on the screen
#plt.savefig('svmplot.pdf') # Save a copy in a f i l e
