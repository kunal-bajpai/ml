from __future__ import division
import numpy as np
import copy

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_extraction import DictVectorizer

import scipy.sparse
import itertools
import sys
sys.path.append("../datasets")
import fetch

def distance(x,y):
	return pairwise_distances(x,y)
	
x,y,trainx,trainy,testx,testy = fetch.getData()

m=2
meannums = dict()
randnum = []
for clss in np.unique(y):
	randnum.append(np.random.randint(x.shape[0]))
means = x[randnum,:].tolil()
means[:,0]=1
means = means.tocsr()

cx = x.tocoo()
ypred = np.zeros(shape=(x.shape[0],))
n=1;
d = np.zeros(shape=(x.shape[0],len(np.unique(y))),dtype=np.float)
u = np.zeros(shape=(x.shape[0],len(np.unique(y))),dtype=np.float)
while True:
	print "Iteration #%d" % (n)
	d = distance(x,means)
	for i in np.arange(x.shape[0]):
		for j in np.unique(y):
			u[i,j] = 1 / np.sum((d[i,j] * (d[i,:]**-1))**(2/(m-1)))
	J = (distance(x,means)**2 * u).sum()
	means = means.tolil()
	for clss in np.unique(y):
		addend = 0
		tmp = scipy.sparse.lil_matrix((x.shape[0],x.shape[0]))
		tmp.setdiag(np.ones(x.shape[0])*(u**m)[:,clss])
		means[clss,:] = (tmp*x).sum(axis=0)/np.sum(u[:,clss]**m)
	Jnew = (distance(x,means)**2 * u).sum()
	ypred = np.argsort(u,axis=1)[:,-1]
	if(J - Jnew)<1:
		break;
	n+=1;
	sys.stdout.write("\033[F")

print "Accuracy is %.2f" %(np.sum(ypred==y)/y.shape[0])
