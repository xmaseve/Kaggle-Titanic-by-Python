# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 13:54:06 2016

@author: YI
"""

import pandas as pd
import numpy as np


def combined_data():
    train = pd.read_csv('C:/Users/YI/Downloads/train.csv')
    test = pd.read_csv('C:/Users/YI/Downloads/test.csv')
    
    label = train['Survived']
    train.drop('Survived',1,inplace=True)
    
    combine = train.append(test)
    combine.reset_index(inplace=True)
    combine.drop('index',inplace=True,axis=1)
    return combine, label
    
combine, label = combined_data()
combine.info()

def get_title(combine):
    combine['Title'] = combine['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
    Title_Dictionary = {"Capt":       "Officer",
                        "Col":        "Officer",
                        "Major":      "Officer",
                        "Jonkheer":   "Royalty",
                        "Don":        "Royalty",
                        "Sir" :       "Royalty",
                        "Dr":         "Officer",
                        "Rev":        "Officer",
                        "the Countess":"Royalty",
                        "Dona":       "Royalty",
                        "Mme":        "Mrs",
                        "Mlle":       "Miss",
                        "Ms":         "Mrs",
                        "Mr" :        "Mr",
                        "Mrs" :       "Mrs",
                        "Miss" :      "Miss",
                        "Master" :    "Master",
                        "Lady" :      "Royalty"}
    combine['Title'] = combine.Title.map(Title_Dictionary)
    combine.drop('Name', axis=1, inplace=True)
    dummies_Title = pd.get_dummies(combine['Title'], prefix='Title')
    combine = pd.concat([combine,dummies_Title],axis=1)
    return combine

combine = get_title(combine)

groups = combine.groupby(['Sex', 'Pclass','Title']).median()

def fillna_age(combine):
    def process(row):
        if row['Sex']=='female' and row['Pclass'] == 1:
            if row['Title'] == 'Miss':
                return 30
            elif row['Title'] == 'Mrs':
                return 45
            elif row['Title'] == 'Officer':
                return 49
            elif row['Title'] == 'Royalty':
                return 39

        elif row['Sex']=='female' and row['Pclass'] == 2:
            if row['Title'] == 'Miss':
                return 20
            elif row['Title'] == 'Mrs':
                return 30

        elif row['Sex']=='female' and row['Pclass'] == 3:
            if row['Title'] == 'Miss':
                return 18
            elif row['Title'] == 'Mrs':
                return 31

        elif row['Sex']=='male' and row['Pclass'] == 1:
            if row['Title'] == 'Master':
                return 6
            elif row['Title'] == 'Mr':
                return 41.5
            elif row['Title'] == 'Officer':
                return 52
            elif row['Title'] == 'Royalty':
                return 40

        elif row['Sex']=='male' and row['Pclass'] == 2:
            if row['Title'] == 'Master':
                return 2
            elif row['Title'] == 'Mr':
                return 30
            elif row['Title'] == 'Officer':
                return 41.5

        elif row['Sex']=='male' and row['Pclass'] == 3:
            if row['Title'] == 'Master':
                return 6
            elif row['Title'] == 'Mr':
                return 26
                
    combine['Age'] = combine.apply(lambda row: process(row) if np.isnan(row['Age']) else row['Age'], axis=1)
    return combine
    
combine = fillna_age(combine)

combine[combine.Fare.isnull()] #find out which one is missing and fill the missing values based on information
combine['Fare'] = combine['Fare'].fillna(7.8958)

def process_embarked(combine):
    combine['Embarked'] = combine['Embarked'].fillna(combine.Embarked.value_counts().index[0])
    dummies_Embarked = pd.get_dummies(combine['Embarked'], prefix='Embarked')
    combine = pd.concat([combine,dummies_Embarked],axis=1)
    combine.drop('Embarked',axis=1,inplace=True)   
    return combine

combine = process_embarked(combine)
    
def sex(x):
    if x == 'male':
        return 1
    if x== 'female':
        return 0
        
combine.Sex = combine.Sex.apply(sex) 

def process_pclass(combine):
    pclass_dummies = pd.get_dummies(combine['Pclass'],prefix="Pclass")
    combine = pd.concat([combine,pclass_dummies],axis=1)
    combine.drop('Pclass',axis=1,inplace=True)
    return combine
    
combine = process_pclass(combine)
    
def process_family(combine):
    combine['FamilySize'] = combine['Parch'] + combine['SibSp'] + 1
    combine['Singleton'] = combine['FamilySize'].map(lambda s : 1 if s == 1 else 0)
    combine['SmallFamily'] = combine['FamilySize'].map(lambda s : 1 if 2<=s<=4 else 0)
    combine['LargeFamily'] = combine['FamilySize'].map(lambda s : 1 if 5<=s else 0)
    return combine
    
combine = process_family(combine)
#Drop cabin because of too many missing values
#Drop ticket because it's related to fare
combine.drop(['PassengerId', 'Cabin', 'Ticket', 'Title', 'FamilySize', 'SibSp', 'Parch'], axis=1, inplace=True) 

def normalize(x):
    return (x-np.mean(x))/np.std(x)

def scale_all_features(combine):
    features = list(combine.columns)
    combine[features] = combine[features].apply(normalize,axis=0)
    return combine
    
combine = scale_all_features(combine)
    
traindata = combine.ix[0:890]
testdata = combine.ix[891:]  
 

def tolist(data):
    new = data.as_matrix()
    #Y = label.as_matrix()
    X = new[:,[2,1,5,0,14]
    m, n = np.shape(X)
    x = np.ones((m,n+1))
    x[:, 1:] = X
    #y = Y.reshape((m,1))
    return x

x= tolist(traindata)
test= tolist(testdata)
        

