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
passengerid = test.PassengerId
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

passengerid = test.PassengerID

##Logistic regression
#sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


#loss function
theta = np.ones((n+1, 1))
def loss(x, y, theta):
    z = x.dot(theta)
    L = -y.T.dot(np.log(sigmoid(z)))-(1-y).T.dot(np.log(1 - sigmoid(z)))
    return L
   
loss(x, y, theta) 

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
    
theta = sga(x, y)

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

result = classify(x_test, theta)

presult=pd.DataFrame(result, columns=['Survived'])
submission = pd.concat([test, presult], axis=1)
submission = submission[['PassengerId','Survived']]
submission.to_csv('C:/Users/YI/Downloads/Titanic_submission.csv')



def classify(x, theta):
    prob = sigmoid(np.sum(x * theta))
    if prob >= 0.5:
        return 1
    else:
        return 0
        
        
def testerror(x, y):
    m, n = np.shape(newtest)
    errorcount = 0
    theta = sga(x, y, 1000)
    for i in range(m):
        if classify(x[i], theta) != y[i]:
            errorcount += 1
    accuracy = 1 - float(errorcount) / m
    return accuracy
    



##Decision Tree
#calculate entropy
def entropy(x):
    m = len(x)
    labelcounts = {}
    for i in x:
        label = i[-1]
        if label not in labelcounts.keys():
            labelcounts[label] = 0
        labelcounts[label] += 1
    entropy = 0
    for key in labelcounts:
        prob = float(labelcounts[key]) / m
        entropy -= prob * np.log(prob, 2)
    return entropy
    

##PCA
def pca(x, k):
    avg = np.mean(x, axis=0)
    x = x - avg
    covx = np.cov(x, rowvar=0)
    eigvalue, eigvec = np.linalg.eig(covx)
    valindex = np.argsort(eigvalue)
    valindex = valindex[:-(valindex+1):-1]
    redeigvec = eigvec[:, valindex]
    lowx = x * redeigvec
    reconx =  lowx * redeigvec.T + x
    return reconx
    





########################################################################
for i in b:
    for j in range(3):
        cluster=[]
        if j == i:
            cluster.append(j)
        print cluster
            
def f():
    return 2

a=f()

def f(x):
    a = []
    while x > 0:
        a.append(x)
        f(x-1)

def Square(x):
    return SquareHelper(abs(x), abs(x))

def SquareHelper(n, x):
    if n == 0:
        return 0
    return SquareHelper(n-1, x) + x

def isPalindrome(aString):
    m = len(aString)
    l = []
    for i in range(m):
        if aString[i] == aString[m-i-1]:
            l.append(1)
        else:
            l.append(0)

    if 0 not in l:
        return True
    else:
        return False


a = [1,2,3]
b = [3,4,5]

l = []
for i,j in zip(a,b):
    l.append( i * j)
print sum(l)

def flatten(aList):
    l =[]
    for i in aList:
        
        if type(i) == 'str':
            l.append(i)
        elif type(i) == 'int':
            l.append(i)
    if type(i) == 'list':
        l.extend(flatten(i))
    
    return l

h=[[1,'a',['cat'],2],[[[3]],'dog'],4,5]

def flatten(lst):
	return sum( ([x] if not isinstance(x, list) else flatten(x)
		     for x in lst), [] ):
           
           
def dict_interdiff(d1, d2):
    # symmetric difference, keys in either d1 or d2 but not both.
    lst = []
    sym_diff = d1.viewkeys() ^ d2
    # intersection, keys that are common to both d1 and d2.
    intersect = d1.viewkeys() & d2
    # apply f on values of the keys that common to both dicts.
    a = {k: f(d1[k], d2[k]) for k in intersect}
    b = {k: d1[k] for k in sym_diff & d1.viewkeys()}
    # add key/value pairings from d2 using keys that appear in sym_diff 
    b.update({k: d2[k] for k in sym_diff & d2.viewkeys()})
    lst.append(a)
    lst.append(b)
    c = tuple(lst)
    return c
                
                
                
         