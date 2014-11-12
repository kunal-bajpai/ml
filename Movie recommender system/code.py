import scipy.io as sp
import numpy as np
import csv

def gradDesc(y,r,alpha=0.1,n_iter=1,lmbda=0):
	x = np.random.random((r.shape[0],100))
	theta = np.random.random((r.shape[1],100))
	for a in np.arange(n_iter):
		xgrad = np.empty(x.shape)
		thetagrad = np.empty(theta.shape)
		xgrad = [np.sum([(np.sum(theta[j] * x[i]) - y[i,j])*theta[j,k] + lmbda * x[i,k] if r[i,j]==1 else 0 for j in np.arange(theta.shape[0])]) for k in np.arange(x.shape[1]) for i in np.arange(x.shape[0])]
		xgrad = np.reshape(np.array(xgrad),x.shape)
		thetagrad = [np.sum([(np.sum(theta[j] * x[i]) - y[i,j])*x[i,k] + lmbda * theta[j,k] if(r[i,j]==1) else 0 for i in np.arange(x.shape[0])]) for k in np.arange(theta.shape[1]) for j in np.arange(theta.shape[0])]
		thetagrad = np.reshape(np.array(thetagrad),theta.shape)
		x -= alpha * xgrad
		theta -= alpha * thetagrad
	return x,theta

print "Importing data..."
data = sp.loadmat("data.mat")
movies = np.genfromtxt("movie_ids.txt",delimiter='\t',dtype=str)
movies = [' '.join(row.split()[1:]) for row in movies]
print "Imported data for %d movies and %d viewers" % data['R'].shape

print "Computing model..."
x,theta = gradDesc(data['Y'],data['R'])
