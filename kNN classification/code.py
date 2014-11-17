from __future__ import division

import sys
sys.path.append('../datasets')
import fetch

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from scipy.stats import mode

x,y,trainx,trainy,testx,testy = fetch.getData()

n_neighbors = int(input("\nPlease enter number of neighbors to use : "))
print "\nPredicting..."
dist = pairwise_distances(testx,trainx)
lowest = np.argsort(dist,axis=1)[:,:n_neighbors]
ypred = np.array([j for j in [int(mode(y[row])[0][0]) for row in lowest]])
print "Accuracy = %.2f\n" % (np.sum((testy==ypred).astype(int))/ypred.shape[0])
