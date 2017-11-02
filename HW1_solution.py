import numpy
from urllib.request import urlopen
import scipy.optimize
import random
from sklearn import svm
import math
from collections import defaultdict

def parseData(fname):
  for l in urlopen(fname):
    yield eval(l)

print("Reading data...")
data = list(parseData("http://jmcauley.ucsd.edu/cse190/data/beer/beer_50000.json"))
print("done")

##################################################
# Helper functions                               #
##################################################

def inner(x, y):
  sum = 0
  for a,b in zip(x,y):
    sum += a*b
  return sum

def squaredDiff(x, y):
  sum = 0
  for a,b in zip(x,y):
    sum += (a-b)*(a-b)
  return sum

##################################################
# Simple statistics                              #
##################################################

# Find the variance of the 'review/taste' value

y = [d['review/taste'] for d in data]

mean = sum(y) * 1.0 / len(y)

mse_test = squaredDiff([mean for d in data], y) / len(y)

print("Variance = " + str(mse_test))

styleCounts = defaultdict(int)

for d in data:
  styleCounts[d['beer/style']] += 1

print(styleCounts)

##################################################
# Regression on style                            #
##################################################

def feature(datum):
  isAIPA = 0
  if datum['beer/style'] == 'American IPA':
    isAIPA = 1
  feat = [1, isAIPA]
  return feat

X = [feature(d) for d in data]
y = [d['review/taste'] for d in data]

theta,residuals,rank,s = numpy.linalg.lstsq(X, y)

print("theta = " + str(theta))

##################################################
# Train/test splits                              #
##################################################

X_train = X[:int(len(X)/2)]
X_test = X[int(len(X)/2):]

y_train = y[:int(len(y)/2)]
y_test = y[int(len(y)/2):]

theta,residuals,rank,s = numpy.linalg.lstsq(X_train, y_train)

predictions_train = [inner(x,theta) for x in X_train]
mse_train = squaredDiff(predictions_test, y_train) / len(y_train)

predictions_test = [inner(x,theta) for x in X_test]
mse_test = squaredDiff(predictions_test, y_test) / len(y_test)

print("Train MSE = " + str(mse_train) + ", test MSE = " + str(mse_test))

##################################################
# Experiments for every style (CSE258)           #
##################################################

commonStyles = [s for s in styleCounts if styleCounts[s] >= 50]
styleInds = dict(zip(list(commonStyles), range(len(commonStyles))))

def feature(datum):
  feat = [1] + [0]*len(styleInds)
  if datum['beer/style'] in styleInds:
    feat[styleInds[datum['beer/style']]] = 1
  return feat

X = [feature(d) for d in data]
X_train = X[:int(len(X)/2)]
X_test = X[int(len(X)/2):]

theta,residuals,rank,s = numpy.linalg.lstsq(X_train, y_train)

predictions_train = [inner(x,theta) for x in X_train]
mse_train = squaredDiff(predictions_test, y_train) / len(y_train)

predictions_test = [inner(x,theta) for x in X_test]
mse_test = squaredDiff(predictions_test, y_test) / len(y_test)

print("Train MSE = " + str(mse_train) + ", test MSE = " + str(mse_test))

##################################################
# Classification                                 #
##################################################

y = [d['beer/style'] == 'American IPA' for d in data]
y_train = y[:int(len(y)/2)]
y_test = y[int(len(y)/2):]

def feature(datum):
  feat = [1, datum['beer/ABV'], datum['review/taste']]
  return feat

X = [feature(d) for d in data]
X_train = X[:int(len(X)/2)]
X_test = X[int(len(X)/2):]

for C in [0.1,10,1000,100000]:
  clf = svm.SVC(C=1000)
  clf.fit(X_train, y_train)
  test_predictions = clf.predict(X_test)
  test_labels = [a > 0 for a in test_predictions]
  test_correct = [(a == b) for a,b in zip(test_labels, y_test)]
  print("C=" + str(C) + ": Accuracy = " + str(sum(test_correct) * 1.0 / len(test_correct)))

# Better feature?
def feature(datum):
  feat = [1, datum['beer/ABV'], datum['review/taste'], 'IPA' in d['review/text'], 'hops' in d['review/text']]
  return feat  

##################################################
# Logistic regression                            #
##################################################

def sigmoid(x):
  return 1.0 / (1 + math.exp(-x))

# NEGATIVE Log-likelihood
def f(theta, X, y, lam):
  loglikelihood = 0
  for i in range(len(X)):
    logit = inner(X[i], theta)
    loglikelihood -= math.log(1 + math.exp(-logit))
    if not y[i]:
      loglikelihood -= logit
  for k in range(len(theta)):
    loglikelihood -= lam * theta[k]*theta[k]
  # for debugging
  print("ll = " + str(loglikelihood))
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

lam = 1.0
theta,_,_ = scipy.optimize.fmin_l_bfgs_b(f, [0]*len(X[0]), fprime, pgtol = 10, args = (X_train, y_train, lam))

test_predictions = [inner(x,theta) for x in X_test]
test_labels = [a > 0 for a in test_predictions]
test_correct = [(a == b) for a,b in zip(test_labels, y_test)]

print("Accuracy = " + str(sum(test_correct) * 1.0 / len(test_correct)))

##################################################
# Validation pipeline                            #
##################################################

X_train = X[:int(len(X)/2)]
y_train = y[:int(len(y)/2)]
X_valid = X[int(len(X)/2):int(3*len(X)/4)]
y_valid = y[int(len(y)/2):int(3*len(y)/4)]
X_test = X[int(3*len(X)/4):]
y_test = y[int(3*len(X)/4):]

for c in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
  clf = svm.SVC(C = c)
  clf.fit(X_train, y_train)
  train_predictions = clf.predict(X_train)
  train_labels = [a > 0 for a in train_predictions]
  train_correct = [(a == b) for a,b in zip(train_labels, y_train)]
  acc_train = sum(train_correct) * 1.0 / len(train_correct)
  valid_predictions = clf.predict(X_valid)
  valid_labels = [a > 0 for a in valid_predictions]
  valid_correct = [(a == b) for a,b in zip(valid_labels, y_valid)]
  acc_valid = sum(valid_correct) * 1.0 / len(valid_correct)
  test_predictions = clf.predict(X_test)
  test_labels = [a > 0 for a in test_predictions]
  test_correct = [(a == b) for a,b in zip(test_labels, y_test)]
  acc_test = sum(test_correct) * 1.0 / len(test_correct)
  print("c = " + str(c) + ";\ttrain=" + str(acc_train) + "; validate=" + str(acc_valid) + "; test=" + str(acc_test))