from __future__ import division

import sys
sys.path.append("../datasets")
import fetch
import numpy as np

from sklearn.metrics import pairwise_distances

x,y,trainx,trainy,testx,testy = fetch.getData()
ypred = np.arange(y.shape[0])
print "\nCalculating distances..."

#Use only with single linkage, complete linkage and group average. Comment out next 2 lines for centroid based
dist = pairwise_distances(x,x)
np.fill_diagonal(dist,np.max(dist))

print "\nClustering..."
while True:
	print "Number of clusters = %d\t" % (len(np.unique(ypred)))
	#dist2 = np.reshape([np.max(dist[ypred==clss,ypred==clss2]) for clss2 in np.nditer(np.unique(ypred)) for clss in np.nditer(np.unique(ypred))],(len(np.unique(ypred)),len(np.unique(ypred)))) #complete linkage
	#dist2 = np.reshape([np.min(dist[ypred==clss,ypred==clss2]) for clss2 in np.nditer(np.unique(ypred)) for clss in np.nditer(np.unique(ypred))],(len(np.unique(ypred)),len(np.unique(ypred)))) #single linkage
	dist2 = np.reshape([np.mean(dist[ypred==clss,ypred==clss2]) for clss2 in np.nditer(np.unique(ypred)) for clss in np.nditer(np.unique(ypred))],(len(np.unique(ypred)),len(np.unique(ypred)))) #group average
	#dist2 = np.reshape([pairwise_distances(x[ypred==clss].mean(axis=0),x[ypred==clss2].mean(axis=0))[0,0] for clss2 in np.nditer(np.unique(ypred)) for clss in np.nditer(np.unique(ypred))],(len(np.unique(ypred)),len(np.unique(ypred)))) #centroid based
	#np.fill_diagonal(dist2,np.max(dist2)) #use with centroid based
	minpair = np.argmin(dist2)
	rownum = (minpair//dist2.shape[1])
	ypred[ypred==rownum] = minpair - ((rownum*dist2.shape[1]) + 1)
	if(len(np.unique(ypred)) == len(np.unique(y))):
		break
