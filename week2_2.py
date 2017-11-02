# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 19:31:34 2017

@author: sand
"""

import numpy
import urllib
import scipy.optimize
import random
from sklearn import svm

def parseData(fname):
  for l in urllib.urlopen(fname):
    yield eval(l)

def feature_supplement(datum):
    feat = []
    feat.append(datum['beer/ABV'])
    feat.append(datum['review/taste'])
    
    if 'amber' in datum['beer/name']:
        feat.append(1)
    else:
        feat.append(0)   
    if 'copper' in datum['beer/name']:
        feat.append(1)
    else:
        feat.append(0) 
    
    
        
    return feat
    #feat.append(datum['review/aroma'])
    #feat.append(datum['review/palate'])
    #feat.append(datum['review/appearance'])
    
    '''if ('IPA' in datum['review/text'] or 'ipa' in datum['review/text']):
        feat.append(1)
    else:
        feat.append(0)

    if 'american' in datum['review/text'] or 'American' in datum['review/text'] :
        feat.append(1)
    else:
        feat.append(0)
        
    if 'IPA' in datum['beer/name'] or 'ipa' in datum['beer/name']:
        feat.append(1)
    else:
        feat.append(0)
        
        
        if 'gold' in datum['beer/name']：
        feat.append(1)
    else:
        feat.append(0)
        

        '''
        
    

    # Add the features of reviews on aroma and palate
    
    #feat.append(datum['review/overall'])   
    #return feat
    # Add the features whether the text review contains 'american' or 'IPA''
    
    
data = list(parseData("http://jmcauley.ucsd.edu/cse190/data/beer/beer_50000.json"))
random.shuffle(data)
y = ["American IPA" in d['beer/style'] for d in data]
X = [feature_supplement(d) for d in data]
#X = [[b['beer/ABV'], b['review/taste']] for b in data]

X_train = X[:25000]
X_test = X[25000:]

y_train = y[:25000]
y_test = y[25000:]

clf = svm.SVC(C=100000, kernel = 'linear')
clf.fit(X_train, y_train)

# Generate the predictions
predictions_train = clf.predict(X_train)
predictions_test = clf.predict(X_test)
# Calculate the accuracy
match_test = [(i == j) for i,j in zip(predictions_train, y_test)]
accuracy_test = sum(match_test) * 1.0 / len(match_test)
match_train = [(i == j) for i,j in zip(predictions_test, y_train)]
accuracy_train = sum(match_train) * 1.0 / len(match_train)