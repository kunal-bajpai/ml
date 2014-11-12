import numpy as np
import scipy.io as sp

from sklearn.svm import SVC
from sklearn import grid_search

print "Importing data..."
data = sp.loadmat("data.mat")
x = data['X']
y = np.ravel(data['y'])
test = sp.loadmat("test.mat")
xtest = test['Xtest']
ytest = np.ravel(test['ytest'])
vocab = np.genfromtxt("vocab.txt",dtype=str)[:,1]
vocab = np.insert(vocab,0,'')
print "Imported %d train, %d test samples with %d features" % (x.shape[0],xtest.shape[0],x.shape[1])
print "Sample processed email : ",vocab[x[0]==1]

print "Training classifier..."
clf = SVC(C=3,gamma=0.01)
clf.fit(x,y)
print "Test set accuracy is %.4f" % (clf.score(xtest,ytest))
