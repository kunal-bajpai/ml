from __future__ import division
import numpy as np
import copy
from sklearn.metrics.pairwise import pairwise_distances

import scipy.sparse
import itertools
import sys
sys.path.append("../datasets")
import fetch

def gaussian(x,y,gamma=0.01):
	return np.exp(-gamma*(pairwise_distances(x,y)**2))
	
x,y,trainx,trainy,testx,testy = fetch.getData()

means = dict()
meannums = dict()
for clss in np.unique(y):
	clsstx = trainx[(trainy==clss).nonzero()[0],:]
	randnum = np.random.randint(clsstx.shape[0])
	means[clss] = clsstx[randnum]
	meannums[clss] = randnum;
print "Random means {class:sample_number} => ",meannums

cx = x.tocoo()
ypred = np.zeros(shape=(x.shape[0],))
j=1;
while True:
	print "Iteration #%d" % (j)
	meansprev = copy.deepcopy(means)
	dist = np.zeros(shape=(x.shape[0],len(means),));
	for clss,mean in means.iteritems():
		dist[:,clss] = gaussian(x,mean).ravel()
	ypred = np.argsort(dist,axis=1)[:,-1]
	for clss in np.unique(y):
		clsstx = trainx[(trainy==clss).nonzero()[0],:]
		means[clss] = scipy.sparse.csr_matrix(clsstx.mean(axis=0))
	brk = True;
	for clss in np.unique(y):
		if(meansprev[clss] - means[clss]).nnz!=0:
			brk = False;
	if(brk):
		break
	j+=1;
	sys.stdout.write("\033[F")

print "Accuracy is %.2f" %(np.sum(ypred==y)/y.shape[0])
