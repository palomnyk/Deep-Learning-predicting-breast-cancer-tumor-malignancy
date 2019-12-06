#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Homework3, Machine Learning Fall 2019

BINF 6210 / BINF 8210:  Machine LearningFinal ProjectDue, Tuesday, December 10, 20191.  (50  points)  Implement  your  own  neural  network  having  two  hidden  layers  inPython.2.  (30 points) Apply your neural network to the Wisconsin Diagnostic Breast Cancer(WDBC) data for classifying the cancer.  Perform leave-one-out cross validation.The data can be downloaded from the UCI (UC Irvine) Machine Learning Repos-itory.  The webpage that describes this breast cancer data repository can be ac-cessed by clicking on this link:Breast Cancer Wisconsin (Diagnostic) Data Setand then clickData Folderat the top of the page to go to the download site.Or you can click the following link directly to go to the download site:DownloadThe data is in filewdbc.dataand description of the data is in filewdbc.names

@author: aaronyerke
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import itertools
import os
from sklearn.model_selection import LeaveOneOut

# Data input and leave-1-out cross validation

# Change directory to where "wdbc.data" is located
os.chdir('/Users/brando/Documents/School/GitWork/Deep-Learning-predicting-breast-cancer-tumor-malignancy/')
dataf = pd.read_csv("wdbc.data",header=None)
data = np.array(dataf) # Create np array



cost_list = []
delta_j_list = []
prediction_list = []

np.set_printoptions(threshold=np.inf)

def plotCf(a,b,t):
    cf =confusion_matrix(a,b)
    plt.imshow(cf,cmap=plt.cm.Blues,interpolation='nearest')
    plt.colorbar()
    plt.title(t)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    tick_marks = np.arange(len(set(a))) # length of classes
    class_labels = ['0','1']
    plt.xticks(tick_marks,class_labels)
    plt.yticks(tick_marks,class_labels)
    thresh = cf.max() / 2.
    for i,j in itertools.product(range(cf.shape[0]),range(cf.shape[1])):
        plt.text(j,i,format(cf[i,j],'d'),horizontalalignment='center',color='white' if cf[i,j] >thresh else 'black')
    plt.show();


def Sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def Relu(Z):
    return np.maximum(0, Z)


def dRelu2(dZ, Z):
    dZ[Z <= 0] = 0
    return dZ


def dRelu(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


def dSigmoid(Z):
    s = 1 / (1 + np.exp(-Z))
    dZ = s * (1 - s)
    return dZ


class dlnet:#neural net
    def __init__(self, x, y):
        self.debug = 0;
        self.X = x
        self.Y = y
        self.Yh = np.zeros((1, self.Y.shape[1]))
        self.dims = [9, 15, 15, 1]
        self.param = {}#weights and biases
        self.ch = {}#cache variable
        self.grad = {}
        self.loss = []
        self.lr = 0.003
        self.sam = self.Y.shape[1]
        self.threshold = 0.5

    def nInit(self):
        np.random.seed(1)
        self.param['W1'] = np.random.randn(self.dims[1], self.dims[0]) / np.sqrt(self.dims[0])
        self.param['b1'] = np.zeros((self.dims[1], 1))
        self.param['W2'] = np.random.randn(self.dims[2], self.dims[1]) / np.sqrt(self.dims[1])
        self.param['b2'] = np.zeros((self.dims[2], 1))
        self.param['W3'] = np.random.randn(self.dims[3], self.dims[2]) / np.sqrt(self.dims[2])
        self.param['b3'] = np.zeros((self.dims[3], 1))
        return

    def forward(self):
        Z1 = self.param['W1'].dot(self.X) + self.param['b1']
        A1 = Relu(Z1)
        self.ch['Z1'], self.ch['A1'] = Z1, A1

        Z2 = self.param['W2'].dot(A1) + self.param['b2']
        A2 = Relu(Z2)
        self.ch['Z2'], self.ch['A2'] = Z2, A2
        
        #print(f"self.param['W3']: {self.param['W3']}, self.param['b3']: {self.param['b3']}")
        Z3 = self.param['W3'].dot(A2) + self.param['b3']
        A3 = Sigmoid(Z3)
        self.ch['Z3'], self.ch['A3'] = Z3, A3

        self.Yh = A3
        loss = self.nloss(A3)
        return self.Yh, loss

    def nloss(self, Yh):
        loss = (1. / self.sam) * (-np.dot(self.Y, np.log(Yh).T) - np.dot(1 - self.Y, np.log(1 - Yh).T))
        return loss

    def backward(self):
        dLoss_Yh = - (np.divide(self.Y, self.Yh) - np.divide(1 - self.Y, 1 - self.Yh))
        
        dLoss_Z3 = dLoss_Yh * dSigmoid(self.ch['Z3'])
        dLoss_A2 = np.dot(self.param["W3"].T, dLoss_Z3)
        dLoss_W3 = 1. / self.ch['A2'].shape[1] * np.dot(dLoss_Z3, self.ch['A2'].T)
        dLoss_b3 = 1. / self.ch['A2'].shape[1] * np.dot(dLoss_Z3, np.ones([dLoss_Z3.shape[1], 1]))

        dLoss_Z2 = dLoss_A2 * dSigmoid(self.ch['Z2'])
        dLoss_A1 = np.dot(self.param["W2"].T, dLoss_Z2)
        dLoss_W2 = 1. / self.ch['A1'].shape[1] * np.dot(dLoss_Z2, self.ch['A1'].T)
        dLoss_b2 = 1. / self.ch['A1'].shape[1] * np.dot(dLoss_Z2, np.ones([dLoss_Z2.shape[1], 1]))

        dLoss_Z1 = dLoss_A1 * dRelu(self.ch['Z1'])
        dLoss_A0 = np.dot(self.param["W1"].T, dLoss_Z1)
        dLoss_W1 = 1. / self.X.shape[1] * np.dot(dLoss_Z1, self.X.T)
        dLoss_b1 = 1. / self.X.shape[1] * np.dot(dLoss_Z1, np.ones([dLoss_Z1.shape[1], 1]))

        self.param["W1"] = self.param["W1"] - self.lr * dLoss_W1
        self.param["b1"] = self.param["b1"] - self.lr * dLoss_b1
        self.param["W2"] = self.param["W2"] - self.lr * dLoss_W2
        self.param["b2"] = self.param["b2"] - self.lr * dLoss_b2
        self.param["W3"] = self.param["W3"] - self.lr * dLoss_W3
        self.param["b3"] = self.param["b3"] - self.lr * dLoss_b3

        return

    def pred(self, x, y):
        self.X = x
        self.Y = y
        comp = np.zeros((1, x.shape[1]))
        pred, loss = self.forward()

        for i in range(0, pred.shape[1]):
            if pred[0, i] > self.threshold:
                comp[0, i] = 1
            else:
                comp[0, i] = 0

        print("Acc: " + str(np.sum((comp == y) / x.shape[1])))

        return comp

    def gd(self, X, Y, iter=3000):
        np.random.seed(1)

        self.nInit()

        for i in range(0, iter):
            Yh, loss = self.forward()
            self.backward()

            if i % 500 == 0:
                #print("Cost after iteration %i: %f" % (i, loss))
                self.loss.append(loss)

        #plt.plot(np.squeeze(self.loss))
        #plt.ylabel('Loss')
        #plt.xlabel('Iter')
        #plt.title("Lr =" + str(self.lr))
        #plt.show()

        return

# Read in the wdbc file
df = pd.read_csv('wdbc.data',header=None)
# Reclassify Benign as 0 an Malignant as 1 in the data file
df.iloc[:,1].replace('B', 0,inplace=True)
df.iloc[:,1].replace('M', 1,inplace=True)
# Verify that all entries can be converted to float
df = df.astype(float)

# Create and store data into a new data frame to scale the enteries
scaled_df=df
names = df.columns[1:13]
scaler = MinMaxScaler()
# Record all rows and columns 2-12 since they are the variables of interest and scale them
scaled_df = scaler.fit_transform(df.iloc[:,1:13])
scaled_df = pd.DataFrame(scaled_df, columns=names)

# Assign all columns except 2 as the independant variables to a panda frame
x=scaled_df.iloc[:,2:13].values.transpose()
# Assign column 2 as the dependent variable to a panda frame
y=df.iloc[:,1].values.transpose()
# Reformat y so that it will not cause an error when run through the ANN
y = np.array([y])


# Empty lists to store predicted vs actual through iterations
y_actual = []
predictions = []
accuracyList = []
accuracy = 0.0
# Iterate through the amount of rows from input data
for i in range(scaled_df.shape[0]):
    # Remove a row based on the index of the loop
    x_train = np.delete(x, i, axis=1)
    y_train = np.delete(y, i, axis=1)

    # Call the ANN class using the training set with hard coded l.r and dims
    nn = dlnet(x_train, y_train)
    nn.lr = 0.01
    nn.dims = [10, 15, 15, 1]
    nn.gd(x_train, y_train, iter=10000)

    # With the row that was removed for the training data test the trained model
    x_test = np.array([np.array(x[:, i])]).transpose()
    y_test = np.array([np.array(y[:, i])]).transpose()

    # Call the pred function from the ANN class and validate that the model is succesfully classifying
    pred_test = nn.pred(x_test, y_test)

    # If the predicted classification is the same add 1 to the accuracy counter
    if pred_test[0][0] == y_test[0][0]:
        accuracy = accuracy + 1.0
    else:
        print('Incorrect Prediction')

    # Append both actual and predicted to seperate ordered lists
    predictions.append(pred_test[0][0])
    y_actual.append(y_test[0][0])

    # Calculate the accuracy of the model so far
    validate = (accuracy / (i + 1.0)) * 100
    accuracyList.append(validate)

    # Print statements to verify model validity
    print("Iteration number: ", i+1)
    print("predictions: ", predictions[i])
    print("y actual: ", y_actual[i])
    print("Model Accuracy: %" + str(validate))
    print('\n')

# Plot the accuracy of the ANN predicted classification vs the number of iterations
plt.plot(accuracyList)
plt.ylabel('% Correct')
plt.xlabel('Iter')
plt.title("% Correct vs Iteration" )
plt.show()

#
# nn.X,nn.Y=x, y
# target=np.around(np.squeeze(y), decimals=0).astype(np.int)
# predicted=np.around(np.squeeze(nn.pred(x,y)), decimals=0).astype(np.int)
# plotCf(target,predicted,'Cf Training Set')
#
# nn.X,nn.Y=xval, yval
# target=np.around(np.squeeze(yval), decimals=0).astype(np.int)
# predicted=np.around(np.squeeze(nn.pred(xval,yval)), decimals=0).astype(np.int)
# plotCf(target,predicted,'Cf Validation Set')
#
# nn.threshold=0.7
#
# nn.X,nn.Y=x, y
# target=np.around(np.squeeze(y), decimals=0).astype(np.int)
# predicted=np.around(np.squeeze(nn.pred(x,y)), decimals=0).astype(np.int)
# plotCf(target,predicted,'Cf Training Set')
#
# nn.X,nn.Y=xval, yval
# target=np.around(np.squeeze(yval), decimals=0).astype(np.int)
# predicted=np.around(np.squeeze(nn.pred(xval,yval)), decimals=0).astype(np.int)
# plotCf(target,predicted,'Cf Validation Set')
#
# nn.threshold=0.9
#
# nn.X,nn.Y=x, y
# target=np.around(np.squeeze(y), decimals=0).astype(np.int)
# predicted=np.around(np.squeeze(nn.pred(x,y)), decimals=0).astype(np.int)
# plotCf(target,predicted,'Cf Training Set')
#
# nn.X,nn.Y=xval, yval
# target=np.around(np.squeeze(yval), decimals=0).astype(np.int)
# predicted=np.around(np.squeeze(nn.pred(xval,yval)), decimals=0).astype(np.int)
# plotCf(target,predicted,'Cf Validation Set')
#
# nn.X,nn.Y=xval, yval
# yvalh, loss = nn.forward()
# print("\ny",np.around(yval[:,0:50,], decimals=0).astype(np.int))
# print("\nyh",np.around(yvalh[:,0:50,], decimals=0).astype(np.int),"\n")