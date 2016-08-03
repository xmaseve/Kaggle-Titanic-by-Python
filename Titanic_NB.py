# -*- coding: utf-8 -*-
"""
Created on Wed Aug 03 17:13:23 2016

@author: YI
"""
import numpy as np
import pandas as pd


train = pd.read_csv('C:/Users/YI/Downloads/train.csv')
train = train.set_index('PassengerId')
train.drop(['Name', 'Ticket', 'Cabin','SibSp', 'Parch', 'Fare'], axis=1, inplace=True)
train = train.fillna(train.mean())
train = train.fillna(train.Embarked.value_counts().index[0])
cutpoint = [0,15,30,60,100]
grouplabel = [0,1,2,3]
cateAge = pd.cut(train.Age, cutpoint, labels=grouplabel)
train['Age'] = cateAge

newtrain = pd.get_dummies(train)
dummyPclass = pd.get_dummies(train.Pclass, prefix='Pclass')
newtrain = pd.concat([newtrain, dummyPclass], axis=1)
newtrain.drop(['Pclass'], axis=1, inplace=True)

new = newtrain.as_matrix()
Y = new[:,0]
X = new[:,1:]


test = pd.read_csv('C:/Users/YI/Downloads/test.csv')
test = test.set_index('PassengerId')
test.drop(['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch','Fare'], axis=1, inplace=True)
test = test.fillna(test.mean())
test['Age'] = pd.cut(test.Age, cutpoint, labels=grouplabel)
newtest = pd.get_dummies(test)
dum_Pclass = pd.get_dummies(newtest.Pclass, prefix='Pclass')
newtest = pd.concat([newtest, dum_Pclass], axis=1)
newtest.drop(['Pclass'], axis=1, inplace=True)
new1 = newtest.as_matrix()

def trainNB(mat, category):
    m, n= np.shape(mat)
    pClass1 = sum(category) / float(m)
    p0Num = np.zeros(n)
    p1Num = np.zeros(n)
    p0Denom = 0.0
    p1Denom = 0.0
    for i in range(m):
        if category[i] == 1:
            p1Num += mat[i]
            p1Denom += 1
        else:
            p0Num += mat[i]
            p0Denom += 1
    p1vec = np.log(p1Num / p1Denom)
    p0vec = np.log(p0Num / p0Denom)
    return p1vec, p0vec, pClass1
   
def classifyNB(vec2Classify, p1vec, p0vec, pClass1):
    m = np.shape(vec2Classify)[0]
    result =[]
    for i in range(m):
        p1 = np.sum(vec2Classify[i] * p1vec) + np.log(pClass1)
        p0 = np.sum(vec2Classify[i] * p0vec) + np.log(1-pClass1)
        if p1 > p0:
            result.append(1)
        else:
            result.append(0)
    return result
        
p1vec, p0vec, pClass1 = trainNB(X, Y)
result = classifyNB(new1, p1vec, p0vec, pClass1)

presult=pd.DataFrame(result, columns=['Survived'])
test = pd.read_csv('C:/Users/YI/Downloads/test.csv')
submission = pd.concat([test, presult], axis=1)
submission = submission[['PassengerId','Survived']]
submission.to_csv('C:/Users/YI/Downloads/Titanic_submission_NB.csv')