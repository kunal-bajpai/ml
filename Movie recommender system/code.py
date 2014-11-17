import scipy.io as sp
import numpy as np
import csv
import sys

def gradDesc(y,r,alpha=0.1,n_iter=100,lmbda=0):
	x = np.random.random((r.shape[0],100))
	theta = np.random.random((r.shape[1],100))
	for a in np.arange(n_iter):
		print "Iteration #%d / %d" % (a,n_iter) 
		xgrad = np.empty(x.shape)
		thetagrad = np.empty(theta.shape)
		cost = np.dot(theta,x.transpose()).transpose() - y
		cost[r==0]=0
		xgrad = np.dot(cost,theta)
		thetagrad = np.dot(cost.transpose(),x)
		x -= alpha * xgrad
		theta -= alpha * thetagrad
		sys.stdout.write("\033[F") #Move cursor up one line
	return x,theta

print "Importing data..."
data = sp.loadmat("data.mat")
movies = np.genfromtxt("movie_ids.txt",delimiter='\t',dtype=str)
movies = [' '.join(row.split()[1:]) for row in movies]
print "Imported data for %d movies and %d viewers" % data['R'].shape

print "Computing model..."
x,theta = gradDesc(data['Y'],data['R'])
