# -*- coding: utf-8 -*-
"""
Created on Mon Jul 04 13:46:29 2016

@author: YI
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv('C:/Users/YI/Downloads/train.csv')
test = pd.read_csv('C:/Users/YI/Downloads/test.csv')

train.info() #to know variables and types and the number of non-missing values
train.describe() #to explore the numeric variables
test.info()
test.describe()

train.Survived.value_counts().plot(kind='bar')
plt.ylabel('frequency')
plt.title('survival(1)')

train.Pclass.value_counts().plot(kind='bar')
plt.ylabel('frequency')
plt.title('The distribution of passengers class')

survived_0 = train.Pclass[train.Survived == 0].value_counts()
survived_1 = train.Pclass[train.Survived == 1].value_counts()
df_survived = pd.DataFrame({'Survived': survived_1, 'Nonsurvived': survived_0})
df_survived.plot(kind='bar', stacked=True)
plt.title('The distribution of survivors based on passengers class')
plt.xlable('Pclass')
plt.ylabel('Frequency')

survived_m = train.Survived[train.Sex == 'male'].value_counts()
survived_f = train.Survived[train.Sex == 'female'].value_counts()
df_survived = pd.DataFrame({'Male': survived_m, 'Female': survived_f})
df_survived.plot(kind='bar', stacked=True)
plt.title('The distribution of survivors based on sex')
plt.xlabel('Sex')
plt.ylabel('Frequency')

avg_male = round(np.mean(train.Age[train.Sex == 'male']))  #calclluate the mean
avg_female = round(np.mean(train.Age[train.Sex == 'female']))

train['Age'][train.Sex == 'male'].fillna(avg_male)   #fill missing values
train['Age'][train.Sex == 'female'].fillna(avg_female)
train = train.fillna(train.mean())
test = test.fillna(test.mean())

#convert categorical variables to numeric variables
dummies_Pclass = pd.get_dummies(train['Pclass'], prefix='Pclass')
dummies_Sex = pd.get_dummies(train['Sex'], prefix='Sex')
dummy_Pclass = pd.get_dummies(test['Pclass'], prefix='Pclass')
dummy_Sex = pd.get_dummies(test['Sex'], prefix='Sex')

#contatenate dummy variables to dateset
newtrain = pd.concat([train, dummies_Pclass, dummies_Sex], axis=1)
newtrain.drop(['Pclass', 'Sex', 'Name', 'Ticket', 'Cabin', 'Embarked', 'Fare', 'SibSp', 'Parch'], axis=1, inplace=True)
newtest = pd.concat([test, dummy_Pclass, dummy_Sex], axis=1)
newtest.drop(['Pclass', 'Sex', 'Name', 'Ticket', 'Cabin', 'Embarked', 'Fare', 'SibSp', 'Parch'], axis=1, inplace=True)


def normalize(x):
    return (x-np.mean(x))/np.std(x)
    
newtrain['Age'] = normalize(newtrain['Age'])
newtest['Age'] = normalize(newtest['Age'])


new = newtrain.as_matrix()
new1 = newtest.as_matrix()

Y = new[:,1]
X = new[:,2:]
m = len(Y)
x = np.ones((m,7))
x[:, 1:] = X
n = len(x[0])  #n=7
y = Y.reshape((m,1))


testX = new1[:,1:]
a = len(new1)
testx = np.ones((a, 7))
testx[:, 1:] = testX



#sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


#loss function
def loss(x, y):
    z = x.dot(theta)
    L = -y.T.dot(np.log(sigmoid(z)))-(1-y).T.dot(np.log(1 - sigmoid(z)))
    return L
   
loss(x, y) 

#batch gradient ascend    
def gradient(x, y, num_iters=200):
    m , n =np.shape(x)
    theta = np.ones((n, 1))    
    alpha = 0.01    
    h = sigmoid(x * theta)
    delta = x.T * (y-h)
    for i in xrange(num_iters):
        theta = theta + alpha * delta
    return theta
     
gradient(x, y, num_iters)  

#SGA stochastic gradient ascend
def sga(x, y, num_iters=200):
    m, n = np.shape(x)
    theta = np.ones(n)
    for i in xrange(num_iters):
        for j in xrange(m):
            alpha = 0.01 + 4 / (1+i+j)
            randindex = int(np.random.uniform(0, m))
            h = sigmoid(np.sum(x[randindex] * theta))
            theta = theta + alpha * (y[randindex] - h) * x[randindex]
    return theta
    
sga(x, y)

def classify(x, theta):
    prob = sigmoid(np.sum(x * theta))
    if prob >= 0.5:
        return 1
    else:
        return 0

        
def testerror(x, y):
    m, n = np.shape(testx)
    errorcount = 0
    theta = sga(x, y, 200)
    for i in range(m):
        if classify(x[i], theta) != y[i]:
            errorcount += 1
    accuracy = 1 - float(errorcount) / m
    return accuracy
    
testerror(x, y)
