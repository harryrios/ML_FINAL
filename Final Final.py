#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Importing the libraries to be used:
import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn import svm
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Importing Data


# In[3]:


df = pd.read_csv('train (1).csv')
df.columns


# In[ ]:


# Preprocessing


# In[4]:


# setting mean age
mean_age = df['Age'].mean() # 29.699...
df['Age'] = df['Age'].fillna(mean_age)


# In[5]:


y = df['Survived']
X = df[['Age', 'SibSp','Parch','Fare']]


# In[6]:


names = df['Name']
titles = ['Mr.', 'Mrs.', 'Miss.', 'Master.', 'Don.', 'Rev.', 'Dr.', 'Mme.', 'Ms.',
       'Major.', 'Lady.', 'Sir.', 'Mlle.', 'Col.', 'Capt.', 'Countess.',
       'Jonkheer.']

is_Mr = []
is_Mrs = []
is_Miss = []
is_Master = []
is_Other = []

for name in names:
    for title in titles:
        if title in name:
            if title == 'Mr.':
                is_Other.append(0)
                is_Mr.append(1)
                is_Mrs.append(0)
                is_Miss.append(0)
                is_Master.append(0)
                
            elif title == 'Mrs.':
                is_Other.append(0)
                is_Mr.append(0)
                is_Mrs.append(1)
                is_Miss.append(0)
                is_Master.append(0)
                
            elif title == 'Miss.':
                is_Other.append(0)
                is_Mr.append(0)
                is_Mrs.append(0)
                is_Miss.append(1)
                is_Master.append(0)
                
            elif title == 'Master.':
                is_Other.append(0)
                is_Mr.append(0)
                is_Mrs.append(0)
                is_Miss.append(0)
                is_Master.append(1)
                
            else:
                is_Other.append(1)
                is_Mr.append(0)
                is_Mrs.append(0)
                is_Miss.append(0)
                is_Master.append(0)
            continue


# In[7]:


is_Mr = pd.Series(is_Mr)
is_Mrs = pd.Series(is_Mrs)
is_Miss = pd.Series(is_Miss)
is_Master = pd.Series(is_Master)
is_Other = pd.Series(is_Other)

X['isMr'] = is_Mr
X['isMrs'] = is_Mrs
X['isMiss'] = is_Miss
X['isMaster'] = is_Master
X['isOther'] = is_Other


# In[8]:


# OHE = One Hot Encoding
OHE_SEX = pd.get_dummies(df.Sex, prefix='Sex')
OHE_PCLASS = pd.get_dummies(df.Pclass, prefix='Pclass')
OHE_EMB = pd.get_dummies(df.Embarked, prefix='Embarked')

X['isMale'] = OHE_SEX['Sex_male']
X['isFemale'] = OHE_SEX['Sex_female']

X['Pclass_1'] = OHE_PCLASS['Pclass_1']
X['Pclass_2'] = OHE_PCLASS['Pclass_2']
X['Pclass_3'] = OHE_PCLASS['Pclass_3']

X['Embarked_Q'] = OHE_EMB['Embarked_Q']
X['Embarked_S'] = OHE_EMB['Embarked_S']
X['Embarked_C'] = OHE_EMB['Embarked_C']

X.head()


# In[9]:


# Finally, we can do Z standarization

y = df['Survived']

scaler = preprocessing.StandardScaler().fit(X)

X = scaler.transform(X)


# In[10]:


# Now let's split our data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[85]:


#  Now that our data is preprocessed, we can begin to run our models
#
#  The first will be Logistical Regression, L1 and L2 with and without polynomial data
#  The second will be SVMs with linear, rbf, and polynomial kernels
#  


# In[ ]:


# Logistical Regression


# In[74]:


# model 1 : L1 logistical regression

def logreg_model(c , X_train, Y_train, X_test, Y_test):
    
    logreg = linear_model.LogisticRegression(penalty = 'l1', C=c, solver = 'saga', max_iter=10000)
    logreg.fit(X_train, Y_train)

    Yhat_train = logreg.predict(X_train) 
    acc_train = logreg.score(X_train, Y_train)
    
    Yhat_test = logreg.predict(X_test)
    acc_test = logreg.score(X_test, Y_test)
    
    print('C = ', c)
    print("Accuracy on training data = %f" % acc_train)
    print("Accuracy on test data = %f" % acc_test)


# In[76]:


cVals = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
for c in cVals:
    logreg_model(c, X_train, Y_train, X_test, Y_test)


# In[77]:


# model 2 : L2 logistical regression

def logreg2_model(c , X_train, Y_train, X_test, Y_test):

    logreg = linear_model.LogisticRegression( C=c, max_iter=10000)
    logreg.fit(X_train, Y_train)

    Yhat_train = logreg.predict(X_train) 
    acc_train = logreg.score(X_train, Y_train)
    
    Yhat_test = logreg.predict(X_test)
    acc_test = logreg.score(X_test, Y_test)
    
    print('C = ', c)
    print("Accuracy on training data = %f" % acc_train)
    print("Accuracy on test data = %f" % acc_test)
    


# In[79]:


cVals = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
for c in cVals:
    logreg2_model(c, X_train, Y_train, X_test, Y_test)


# In[ ]:


# Try again with polynomial transformation


# In[81]:


poly = PolynomialFeatures(2)
X_transformed_train = poly.fit_transform(X_train)
X_transformed_test = poly.fit_transform(X_test)


# In[82]:


cVals = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
for c in cVals:
    logreg_model(c, X_transformed_train, Y_train, X_transformed_test, Y_test)


# In[83]:


cVals = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
for c in cVals:
    logreg2_model(c, X_transformed_train, Y_train, X_transformed_test, Y_test)


# In[ ]:


# Support Vector Machines


# In[86]:


def svm_linear(c, X_train, Y_train, X_test, Y_test):
    
    svc_linear = svm.SVC(probability = False, kernel = 'linear', C = c)
    svc_linear.fit(X_train, Y_train)
    
    Yhat_svc_linear_train = svc_linear.predict(X_train)
    
    acc_train = svc_linear.score(X_train, Y_train)
    acc_test = svc_linear.score(X_test, Y_test)
    
    print('C = ', c)
    print('Train Accuracy = {0:f}'.format(acc_train))
    print('Test Accuracy = {0:f}'.format(acc_test))


# In[87]:


cVals = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
for c in cVals:
    svm_linear(c, X_train, Y_train, X_test, Y_test)


# In[74]:


def svm_rbf(c, X_train, Y_train, X_test, Y_test):
    print('C = ', c)

    svc_rbf = svm.SVC(probability = False, kernel = 'rbf', C = c)
    
    svc_rbf.fit(X_train, Y_train)
    
    acc_train = svc_rbf.score(X_train,Y_train)
    print('Train Accuracy = {0:f}'.format(acc_train))

    acc_test = svc_rbf.score(X_test,Y_test)
    print('Test Accuracy = {0:f}'.format(acc_test))


# In[75]:


cVals = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
for c in cVals:
    svm_rbf(c, X_train, Y_train, X_test, Y_test)


# In[93]:


def svm_polynomial(c, X_train, Y_train, X_test, Y_test):
    print('C = ', c)
    
    svc_polynomial = svm.SVC(probability = False, kernel = 'poly', C = c) 
    svc_polynomial.fit(X_train, Y_train)

    Yhat_svc_poly_train = svc_polynomial.predict(X_train)
    acc_train = svc_polynomial.score(X_train, Y_train)
    
    print('Train Accuracy = {0:f}'.format(acc_train))
    
    Yhat_svc_poly_test = svc_polynomial.predict(X_test)
    acc_test = svc_polynomial.score(X_test, Y_test)

    print('Test Accuracy = {0:f}'.format(acc_test))


# In[94]:


cVals = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
for c in cVals:
    svm_polynomial(c, X_train, Y_train, X_test, Y_test)


# In[ ]:


# Neural Networks


# In[1]:


from sklearn.neural_network import MLPClassifier


# In[23]:


clf = MLPClassifier(solver = 'lbfgs', max_iter = 500, hidden_layer_sizes = (25))
clf.fit(X_train, Y_train)

train_score = clf.score(X_train, Y_train)
test_score = clf.score(X_test, Y_test)

print(train_score)
print(test_score)

# with clf = MLPClassifier(solver = 'lbfgs')
# train = .93
# test = .77


# 1. How about larger alpha?
#---------------------------------
# with clf = MLPClassifier(solver = 'lbfgs', max_iter = 500, alpha = .001)
# train = .95
# test = .77

# with clf = clf = MLPClassifier(solver = 'lbfgs', max_iter = 500, alpha = .01)
# train = .95
# test = .77

# 2. How about less hidden neurons?
#-------------------------------------
# with clf = MLPClassifier(solver = 'lbfgs', hidden_layer_sizes = (50))
# train = .92
# test = .78

# with clf = MLPClassifier(solver = 'lbfgs', hidden_layer_sizes = (25))
# train = .91
# test = .79

# with clf = MLPClassifier(solver = 'lbfgs', hidden_layer_sizes = (10))
# train = .89
# test = .78


# In[35]:


clf = MLPClassifier(solver = 'sgd', hidden_layer_sizes = (10))
clf.fit(X_train, Y_train)

train_score = clf.score(X_train, Y_train)
test_score = clf.score(X_test, Y_test)

print(train_score)
print(test_score)


# with clf = MLPClassifier(solver = 'sgd')
# train = .86
# test  = .82


# 1. How about larger alpha?
#---------------------------------
# with clf = MLPClassifier(solver = 'sgd', alpha = .001)
# train = .84
# test  = .82

# with clf = MLPClassifier(solver = 'sgd', alpha = .01)
# train = .84
# test  = .81


# 2. How about less hidden neurons?
#-------------------------------------
# with clf = MLPClassifier(solver = 'sgd', hidden_layer_sizes = (50))
# train = .83
# test  = .82

# with clf = MLPClassifier(solver = 'sgd', hidden_layer_sizes = (25))
# train = .85
# test  = .82

# with clf = MLPClassifier(solver = 'sgd', hidden_layer_sizes = (10))
# train = .82
# test  = .81


# In[73]:


clf = MLPClassifier(solver = 'adam', hidden_layer_sizes = (10), max_iter = 500)
clf.fit(X_train, Y_train)

train_score = clf.score(X_train, Y_train)
test_score = clf.score(X_test, Y_test)

print(train_score)
print(test_score)

# with clf = MLPClassifier(solver = 'adam')
# train = .86
# test  = .82


# 1. How about larger alpha?
#---------------------------------
# with clf = MLPClassifier(solver = 'adam', alpha = .001)
# train = .86
# test  = .82

# with clf = MLPClassifier(solver = 'adam', alpha = .01)
# train = .86
# test  = .82


# 2. How about less hidden neurons?
#-------------------------------------
# with clf = MLPClassifier(solver = 'adam', hidden_layer_sizes = (50))
# train = .85
# test  = .82

# with clf = MLPClassifier(solver = 'adam', hidden_layer_sizes = (25))
# train = .85
# test  = .82

# with clf = MLPClassifier(solver = 'adam', hidden_layer_sizes = (10))
# train = .83
# test  = .85


# In[ ]:


# From these permutations, it seems like 'adam' solver is the best
# According to the scikit Learn website, it is good for large data sets (1k or more examples)

# Additionally, changing alpha seems to be pretty useless, but changing the hiddel layers might
# be helpful.

# In the next round of testing, let's try using 'adam' with different hidden layers and different
# acitvation functions.

# The default solver is ReLu, so going forward let's try identity, logistic, and tanh


# In[41]:


clf = MLPClassifier(solver = 'adam', activation = 'identity', hidden_layer_sizes = (10))
clf.fit(X_train, Y_train)

train_score = clf.score(X_train, Y_train)
test_score = clf.score(X_test, Y_test)

print(train_score)
print(test_score)

# with clf = MLPClassifier(solver = 'adam', activation = 'identity')
# train = .83
# test  = .82


# 1. How about less hidden neurons?
#-------------------------------------
# with clf = MLPClassifier(solver = 'adam', activation = 'identity', hidden_layer_sizes = (50))
# train = .83
# test  = .82

# with clf = MLPClassifier(solver = 'adam', activation = 'identity', hidden_layer_sizes = (25))
# train = .84
# test  = .82

# with clf = MLPClassifier(solver = 'adam', activation = 'identity', hidden_layer_sizes = (10))
# train = .83
# test  = .82


# In[45]:


clf = MLPClassifier(solver = 'adam', activation = 'logistic', hidden_layer_sizes = (10))
clf.fit(X_train, Y_train)

train_score = clf.score(X_train, Y_train)
test_score = clf.score(X_test, Y_test)

print(train_score)
print(test_score)

# with clf = MLPClassifier(solver = 'adam', activation = 'logistic')
# train = .83
# test  = .82


# 1. How about less hidden neurons?
#-------------------------------------
# with clf = MLPClassifier(solver = 'adam', activation = 'logistic', hidden_layer_sizes = (50))
# train = .83
# test  = .83

# with clf = MLPClassifier(solver = 'adam', activation = 'logistic', hidden_layer_sizes = (25))
# train = .83
# test  = .83

# with clf = MLPClassifier(solver = 'adam', activation = 'logistic', hidden_layer_sizes = (10))
# train = .83
# test  = .83


# In[49]:


clf = MLPClassifier(solver = 'adam', activation = 'tanh', hidden_layer_sizes = (10))
clf.fit(X_train, Y_train)

train_score = clf.score(X_train, Y_train)
test_score = clf.score(X_test, Y_test)

print(train_score)
print(test_score)

# with clf = MLPClassifier(solver = 'adam', activation = 'tanh')
# train = .85
# test  = .82


# 1. How about less hidden neurons?
#-------------------------------------
# with clf = MLPClassifier(solver = 'adam', activation = 'tanh', hidden_layer_sizes = (50))
# train = .85
# test  = .82

# with clf = MLPClassifier(solver = 'adam', activation = 'tanh', hidden_layer_sizes = (25))
# train = .85
# test  = .82

# with clf = MLPClassifier(solver = 'adam', activation = 'tanh', hidden_layer_sizes = (10))
# train = .84
# test  = .83


# In[ ]:


# After looking at these, the best is still:
#   clf = MLPClassifier(solver = 'adam', hidden_layer_sizes = (10))

# then using these parameters, what if we changed the number of hidden layers?


# In[51]:


clf = MLPClassifier(solver = 'adam', hidden_layer_sizes = (10))
clf.fit(X_train, Y_train)

train_score = clf.score(X_train, Y_train)
test_score = clf.score(X_test, Y_test)

print(train_score)
print(test_score)

# with clf = MLPClassifier(solver = 'adam', hidden_layer_sizes = (10, 5))
# train = .85
# test  = .83

