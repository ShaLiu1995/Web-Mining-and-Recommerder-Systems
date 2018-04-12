# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 11:13:49 2017

@author: sand
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 10:52:07 2017

@author: sand
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 16:16:59 2017

@author: sand
"""

import numpy
import urllib
import scipy.optimize
import random
from math import exp
from math import log

def parseData(fname):
  for l in urllib.urlopen(fname):
    yield eval(l)

print("Reading data...")
data = list(parseData("http://jmcauley.ucsd.edu/cse190/data/beer/beer_50000.json"))
print("done")

import string
from collections import defaultdict
def word_count(str):
    exclude = string.punctuation + string.digits
    words = str.translate(None, exclude)
    words = words.lower()
    words = words.split()
    
    wordCounts = defaultdict(int)
    for word in words:
          wordCounts[word] += 1
            
    return wordCounts

feature_words = ["lactic","tart", "sour", "citric", "sweet", "acid", "hop", "fruit", "salt", "spicy" ]
def feature(datum, feature_words):
    wordCounts = word_count(datum['review/text'])
    feat = [1]
    for word in feature_words:
        if word in wordCounts:
            feat.append(wordCounts[word])
        else:
            feat.append(0)
    return feat

X = [feature(d, feature_words) for d in data]
y = [d['beer/ABV'] >= 6.5 for d in data]

def inner(x,y):
  return sum([x[i]*y[i] for i in range(len(x))])

def sigmoid(x):
  return 1.0 / (1 + exp(-x))

##################################################
# Logistic regression by gradient ascent         #
##################################################

# NEGATIVE Log-likelihood
def f(theta, X, y, lam):
  loglikelihood = 0
  n = len(y)
  n1 = sum([i == 1 for i in y])
  n2 = sum([i == 0 for i in y])
  for i in range(len(X)):
    logit = inner(X[i], theta)
    if y[i]:
        loglikelihood -= log(1 + exp(-logit)) * n/(2*n1)
    if not y[i]:
        loglikelihood -= log(1 + exp(-logit)) * n/(2*n2)
        loglikelihood -= logit * n/(2*n2)
  for k in range(len(theta)):
    loglikelihood -= lam * theta[k] * theta[k]
  # for debugging
  print("ll =" + str(loglikelihood))
  return -loglikelihood

# NEGATIVE Derivative of log-likelihood
def fprime(theta, X, y, lam):
  dl = [0]*len(theta)
  
  n1 = sum([i == 1 for i in y])
  n2 = sum([i == 0 for i in y])
  n = len(y)
  for i in range(len(X)):
    logit = inner(X[i], theta)
    for k in range(len(theta)):
        if y[i]:
            dl[k] += X[i][k] * (1 - sigmoid(logit)) * n/(2*n1)
        if not y[i]:
            dl[k] += X[i][k] * (1 - sigmoid(logit)) * n/(2*n2)
            dl[k] -= X[i][k] * n/(2*n2)
  for k in range(len(theta)):
    dl[k] -= lam * 2 * theta[k]
  return numpy.array([-x for x in dl])

X_train = X[:len(X)/3]
y_train = y[:len(y)/3]

X_validate = X[len(X)/3 : 2*len(X)/3]
y_validate = y[len(y)/3 : 2*len(y)/3]

X_test = X[2*len(X)/3:]
y_test = y[2*len(y)/3:]


##################################################
# Train                                          #
##################################################

def train(lam):
  theta,_,_ = scipy.optimize.fmin_l_bfgs_b(f, [0]*len(X[0]), fprime, pgtol = 10, args = (X_train, y_train, lam))
  return theta

##################################################
# Predict                                        #
##################################################

def performance(theta, X, y):
  scores = [inner(theta,x) for x in X]
  predictions = [s > 0 for s in scores]
  correct = [(a==b) for (a,b) in zip(predictions,y)]
  acc = sum(correct) * 1.0 / len(correct)
  return acc


##################################################
# Validation pipeline                            #
##################################################

lam = 1.0
theta = train(lam)
print("Accuracy on training set:")
acc_train = performance(theta, X_train, y_train)
print("lambda = " + str(lam) + ":\taccuracy=" + str(acc_train))
print("Accuracy on validation set:")
acc_validate = performance(theta, X_validate, y_validate)
print("lambda = " + str(lam) + ":\taccuracy=" + str(acc_validate))
print("Accuracy on test set:")
acc_test = performance(theta, X_test, y_test)
print("lambda = " + str(lam) + ":\taccuracy=" + str(acc_test))

test_predictions = [inner(x,theta) for x in X_test]
test_labels = [a > 0 for a in test_predictions]
test_correct = [(a == b) for a,b in zip(test_labels, y_test)]

TP = sum([y_test[i] == 1 and test_labels[i] == 1 for i in xrange(len(y_test)-1)])
FN = sum([y_test[i] == 1 and test_labels[i] == 0 for i in xrange(len(y_test)-1)])
TN = sum([y_test[i] == 0 and test_labels[i] == 0 for i in xrange(len(y_test)-1)])
FP = sum([y_test[i] == 0 and test_labels[i] == 1 for i in xrange(len(y_test)-1)])
TPR = (float(TP) / (TP + FN))
TNR = (float(TN) / (TN + FP))
BER = 1 - 0.5*(TPR + TNR)
