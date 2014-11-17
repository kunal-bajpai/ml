from __future__ import division

import numpy as np
import scipy.sparse
import itertools

import sys
sys.path.append("../datasets")
import fetch

x,y,trainx,trainy,testx,testy = fetch.getData()

prior = dict()
total = trainx.shape[0]
condprob = dict()
for term in np.arange(trainx.shape[1]):
	condprob[term] = dict()
print np.unique(y)
print "Training classifier..."
for i,clss in enumerate(np.unique(y),start=1):
	print "\tClass %d / %d" % (i,len(np.unique(y)))
	txclss = trainx[(trainy==clss).nonzero()[0],:]
	prior[clss] = txclss.shape[0]/total
	tfclsstot = txclss.sum() + txclss.shape[1]
	for term in np.arange(trainx.shape[1]):
		print "\tTerm %d / %d" % (term,trainx.shape[1])
		condprob[term][clss] = (txclss[:,term].sum()+1)/tfclsstot
		sys.stdout.write("\033[F")
	sys.stdout.write("\033[F")

print "\n\n\nPredicting..."
testx = testx.tocoo()
score = dict()
ypred = np.zeros(shape=(testx.shape[0],))
for i in itertools.izip(testx.row):
	score[i] = dict()
for i,doc in enumerate(itertools.izip(testx.row),start=1):
	print "Test doc %d / %d" % (i,testx.shape[0])
	for clss in np.unique(y):
		score[doc][clss] = np.log(prior[clss])
		for term in itertools.izip(testx.col):
			score[doc][clss] += np.log(condprob[term[0]][clss])
	ypred[doc] = max(score[doc].iterkeys(), key=(lambda key: score[doc][key]))
	print "Live accuracy = %.2f" % (np.sum((y[:i]==ypred[:i]).astype(int))/i)
	sys.stdout.write("\033[F")
	sys.stdout.write("\033[F")
