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

def feature(datum):
  feat = [1, datum['review/taste'], datum['review/appearance'], datum['review/aroma'], datum['review/palate'], datum['review/overall']]
  return feat

X = [feature(d) for d in data]
y = [d['beer/ABV'] >= 6.5 for d in data]
#y = [d['beer/style'] == "American IPA" for d in data]

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
  for i in range(len(X)):
    logit = inner(X[i], theta)
    loglikelihood -= log(1 + exp(-logit))
    if not y[i]:
      loglikelihood -= logit
  for k in range(len(theta)):
    loglikelihood -= lam * theta[k]*theta[k]
  # for debugging
  # print("ll =" + str(loglikelihood))
  return -loglikelihood

# NEGATIVE Derivative of log-likelihood
def fprime(theta, X, y, lam):
  dl = [0]*len(theta)
  for i in range(len(X)):
    logit = inner(X[i], theta)
    for k in range(len(theta)):
      dl[k] += X[i][k] * (1 - sigmoid(logit))
      if not y[i]:
        dl[k] -= X[i][k]
  for k in range(len(theta)):
    dl[k] -= lam*2*theta[k]
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
