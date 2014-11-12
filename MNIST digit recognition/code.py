from __future__ import division
import scipy.io as sp
import numpy as np

from sklearn.svm import SVC
from sklearn import cross_validation,grid_search,ensemble,decomposition,neighbors
from pylab import imshow,cm,show

print "Importing data..."
data = sp.loadmat("data.mat")
x = np.array(data['X'],dtype='>d') #dtype is to fix an error that pops saying byte order is not native
y = np.ravel(data['y'])
print "Imported %d samples with %d features\n" % (x.shape[0],x.shape[1])

for i,j in enumerate(np.random.randint(0,x.shape[0],10)):
	if i==0:
		sample = np.reshape(x[j],(20,20),order='F')
	else:
		sample = np.concatenate([sample,np.reshape(x[j],(20,20),order='F')],axis=1)
imshow(sample,cmap=cm.gray)
show()

print "Performing feature selection by PCA..."
pca = decomposition.PCA(n_components=0.99)
x = pca.fit_transform(x)
print "Chosen %d features\n"%(x.shape[1])

print "Evaluating SVM classifier..."
clf = grid_search.GridSearchCV(SVC(),cv=cross_validation.StratifiedKFold(y,n_folds=5),scoring='accuracy',param_grid={'gamma':[0.1],'C':[3]})
clf.fit(x,y)
print "Accuracy is ",clf.grid_scores_
