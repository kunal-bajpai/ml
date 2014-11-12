import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn.neighbors import KernelDensity

print("Importing data...")
train = np.genfromtxt("train.csv",delimiter=',')
y = np.genfromtxt("trainLabels.csv")
print "Imported %d samples with %d features" % train.shape

kde = KernelDensity(kernel='gaussian',bandwidth=0.25).fit(train[:,0,None])
log_dens=kde.score_samples(np.linspace(-4,4,100)[:,None])
plt.plot(np.linspace(-4,4,100),np.exp(log_dens))
plt.show()

"""clf = SVC(C=3,gamma=0.01)
print "Training classifier..."
clf.fit(train,y)
print "Accuracy is ",np.mean(cross_validation.cross_val_score(clf,train,y,cv=5,scoring='accuracy'))"""
