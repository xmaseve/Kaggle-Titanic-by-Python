# -*- coding: utf-8 -*-
"""
Created on Mon Jul 04 13:46:29 2016

@author: YI
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv('C:/Users/YI/Downloads/train.csv')

train.info() #to know variables and types and the number of non-missing values
train.describe() #to explore the numeric variables

train = train.set_index('PassengerId')

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

train = train.fillna(train.mean())

train.groupby('Embarked').Survived.value_counts()
train[train.Embarked.isnull()]  #return missing values of Embark
train = train.fillna(train.Embarked.value_counts().index[0])


#convert categorical variables to numeric variables
def sex(x):
    if x == 'male':
        return 1
    if x== 'female':
        return 0
        
train.Sex = train.Sex.apply(sex)
dummies_Embarked = pd.get_dummies(train['Embarked'], prefix='Embarked')

#contatenate dummy variables to dateset
newtrain = pd.concat([train, dummies_Embarked], axis=1)
newtrain.drop(['Name', 'Ticket', 'Cabin', 'Embarked', 'SibSp', 'Parch'], axis=1, inplace=True)


new = newtrain.as_matrix()

Y = new[:,0]
X = new[:,1:]
m, n = np.shape(X)
x = np.ones((m,n+1))
x[:, 1:] = X
y = Y.reshape((m,1))

#Test dataset
test = pd.read_csv('C:/Users/YI/Downloads/test.csv')
test.info()
test.describe()
test = test.set_index('PassengerId')
test = test.fillna(test.mean())
test.Sex = test.Sex.apply(sex)
dummy_Embarked = pd.get_dummies(test['Embarked'], prefix='Embarked')
newtest = pd.concat([test, dummy_Embarked], axis=1)
newtest.drop(['Name', 'Ticket', 'Cabin', 'Embarked', 'SibSp', 'Parch'], axis=1, inplace=True)
new1 = newtest.as_matrix()
a, b = np.shape(new1)
x_test = np.ones((a,b+1))
x_test[:, 1:] = new1



##Logistic regression
#sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))



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
def sga(x, y, num_iters=500):
    m, n = np.shape(x)
    theta = np.ones(n)
    for i in xrange(num_iters):
        for j in xrange(m):
            alpha = 0.01 + 4 / (1+i+j)
            randindex = int(np.random.uniform(0, m))
            h = sigmoid(np.sum(x[randindex] * theta))
            theta = theta + alpha * (y[randindex] - h) * x[randindex]
    return theta

#SGA with Regularization L2:    
def sga(x, y, lamda, num_iters=500):
    m, n = np.shape(x)
    theta = np.ones(n)
    for i in range(num_iters):
        for j in range(n):
        #for j in xrange(m):
            alpha = 0.01 + 4 / (1+i+j)
            randindex = int(np.random.uniform(0, m))
            h = sigmoid(np.sum(x[randindex] * theta))
            if j == 0:
                theta[j] = theta[j] + alpha * ((y[randindex] - h) * x[randindex,0])
            else:
                theta[j] = theta[j] + alpha * ((y[randindex] - h) * x[randindex,j] + lamda * theta[j])
    return theta
    
    
#Generate predicted values
def classify(x, theta):
    result = []
    m, n = np.shape(x_test)
    for i in range(m):
        prob = sigmoid(np.sum(x[i] * theta))
        if prob >= 0.5:
            result.append(1)
        else:
            result.append(0)
    return result

theta = sga(x, y)
result = classify(x_test, theta)

presult=pd.DataFrame(result, columns=['Survived'])
submission = pd.concat([test, presult], axis=1)
submission = submission[['PassengerId','Survived']]
submission.to_csv('C:/Users/YI/Downloads/Titanic_submission.csv')







    

    


