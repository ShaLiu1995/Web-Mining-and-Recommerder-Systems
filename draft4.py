# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 17:25:38 2017

@author: sand
"""

import numpy as np
import urllib
import scipy.optimize
import random
from sklearn.decomposition import PCA
from collections import defaultdict

def parseData(fname):
    for l in urllib.urlopen(fname):
        yield eval(l)
        
print "Reading data..."
data = list(parseData("http://jmcauley.ucsd.edu/cse190/data/beer/beer_50000.json"))
print "done"

def feature(datum):
    s = datum['review/text'].lower()
    feat = []
    feat.append(s.count('lactic')) 
    feat.append(s.count('tart')) 
    feat.append(s.count('sour')) 
    feat.append(s.count('citric')) 
    feat.append(s.count('sweet')) 
    feat.append(s.count('acid')) 
    feat.append(s.count('hop')) 
    feat.append(s.count('fruit'))
    feat.append(s.count('salt')) 
    feat.append(s.count('spicy')) 
    return feat

X = [feature(d) for d in data]

pca = PCA(n_components = 2)
pca.fit(X)
phi =  pca.components_

X = np.array(X)
X = X.T

y =  np.mat(phi) * np.mat(X)
y = np.array(y)

import matplotlib.pyplot as plt
plt.scatter(y[0][0],y[1][0],color = 'b')
plt.scatter(y[0][1],y[1][1],color = 'r')
plt.show()