# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 14:05:07 2016

@author: YI
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 21:00:12 2016

@author: YI
"""

import numpy as np
import pandas as pd

train = pd.read_csv('C:/Users/YI/Downloads/train.csv')
train = train.set_index('PassengerId')
train.info()
train.drop(['Name', 'Ticket', 'Cabin','Fare'], axis=1, inplace=True)
train = train.fillna(train.mean())
train = train.fillna(train.Embarked.value_counts().index[0])

newtrain = pd.get_dummies(train)
dummyPclass = pd.get_dummies(train.Pclass, prefix='Pclass')
newtrain = pd.concat([newtrain, dummyPclass], axis=1)
newtrain.drop(['Pclass'], axis=1, inplace=True)

label = newtrain['Survived']
label = label.values.tolist()
newtrain.drop(['Survived'], axis=1, inplace=True)
dataset = newtrain.values.tolist()

#test dataset
test = pd.read_csv('C:/Users/YI/Downloads/test.csv')
test = test.set_index('PassengerId')
test.drop(['Name', 'Ticket', 'Cabin','Fare'], axis=1, inplace=True)
test = test.fillna(train.mean())
test = test.fillna(train.Embarked.value_counts().index[0])

newtest = pd.get_dummies(test)
dummy_Pclass = pd.get_dummies(test.Pclass, prefix='Pclass')
newtest = pd.concat([newtest, dummy_Pclass], axis=1)
newtest.drop(['Pclass'], axis=1, inplace=True)
testdataset = newtest.values.tolist()
   
# Building weak stump function
def buildWeakStump(dataset,label,D):
    dataMatrix = np.mat(dataset)
    labelmatrix = np.mat(label).T
    m,n = np.shape(dataMatrix)
    numstep = 10.0
    bestStump = {}
    bestClass = np.mat(np.zeros((m,1)))
    minErr = np.inf
    for i in range(n):
        datamin = dataMatrix[:,i].min()
        datamax = dataMatrix[:,i].max()
        stepSize = (datamax - datamin) / numstep
        for j in range(-1,int(numstep)+1):
            for inequal in ['lt','gt']:
                threshold = datamin + float(j) * stepSize
                predict = stumpClassify(dataMatrix,i,threshold,inequal)
                err = np.mat(np.ones((m,1)))
                err[predict == labelmatrix] = 0
                weighted_err = D.T * err;
                if weighted_err < minErr:
                    minErr = weighted_err
                    bestClass = predict.copy()
                    bestStump['dim'] = i
                    bestStump['threshold'] = threshold
                    bestStump['ineq'] = inequal
    return bestStump, minErr, bestClass

# Use the weak stump to classify training data
def stumpClassify(datamat,dim,threshold,inequal):
    res = np.ones((np.shape(datamat)[0],1))
    if inequal == 'lt':
        res[datamat[:, dim] <= threshold] = 0   #Maybe it is -1 based on some situations
    else:
        res[datamat[:, dim] > threshold] = 0
    return res

# Training
def AdaboostTrain(dataset,label,numIt = 10):
    weakClassifiers = []
    m = np.shape(dataset)[0]
    D = np.mat(np.ones((m,1))/m)
    EnsembleClassEstimate = np.mat(np.zeros((m,1)))
    for i in range(numIt):
        bestStump, error, classEstimate = buildWeakStump(dataset,label,D)
        alpha = float(0.5*np.log((1.0-error) / max(error, 1e-15)))
        bestStump['alpha'] = alpha
        weakClassifiers.append(bestStump)
        weightD = np.multiply((-1*alpha*np.mat(label)).T,classEstimate)
        D = np.multiply(D,np.exp(weightD))
        D = D/D.sum()
        EnsembleClassEstimate += alpha*classEstimate
        EnsembleErrors = np.multiply(np.sign(EnsembleClassEstimate)!=np.mat(label).T,\
                                  np.ones((m,1)))  
        errorRate = EnsembleErrors.sum()/m
        print "total error:  ",errorRate
        if errorRate == 0.0:
            break
    return weakClassifiers

classifier = AdaboostTrain(dataset,label,5)

# Applying adaboost classifier for a single data sample
def adaboostClassify(test,classifier):
    datamatrix = np.mat(test)
    m = np.shape(datamatrix)[0]
    EnsembleClassEstimate = np.mat(np.zeros((m,1)))
    for i in range(len(classifier)):
        classEstimate = stumpClassify(datamatrix,classifier[i]['dim'],classifier[i]['threshold'],classifier[i]['ineq'])
        EnsembleClassEstimate += classifier[i]['alpha']*classEstimate
    return np.sign(EnsembleClassEstimate)


# Testing
def test(test,classifier):
    m = np.shape(test)[0]
    result = []
    for i in range(m):
        result.append(adaboostClassify(test[i],classifier))
    return result
    
classifier = AdaboostTrain(dataset,label,5) 
adaboostClassify(testdataset[0],classifier)    
result = test(testdataset, classifier)
result = np.reshape(result,(418,1))
presult=pd.DataFrame(result, columns=['Survived'])
presult.to_csv('C:/Users/YI/Downloads/Titanic_submission_Adaboost.csv')