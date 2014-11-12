from __future__ import division
import numpy as np
import scipy.io as sp
import matplotlib.pyplot as plt

from sklearn import covariance

print "Importing data..."
data = sp.loadmat("data.mat")
x = np.array(data['X'])
xval = np.array(data['Xval'])
yval = np.ravel(data['yval']).astype(int)
plt.plot(x[:,0],x[:,1],'rx')
plt.xlabel("Latency (ms)")
plt.ylabel("Throughput (mb/s)")
plt.title("Training set")
plt.show()

plt.clf()
plt.plot(xval[:,0][yval==1],x[:,1][yval==1],'rx',xval[:,0][yval==0],x[:,1][yval==0],'gx')
plt.xlabel("Latency (ms)")
plt.ylabel("Throughput (mb/s)")
plt.title("Cross validation set")
plt.legend(('outlier','inlier'),scatterpoints=1)
plt.show()

yval[yval==1] = -1
yval[yval==0] = 1
det = covariance.EllipticEnvelope(contamination=0.03)
det.fit(x)
print "Accuracy is %.4f" % (det.score(xval,yval))
