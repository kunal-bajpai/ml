import numpy as np
import scipy.io as sp
import matplotlib.pyplot as plt

from sklearn import pipeline,linear_model,preprocessing,cross_validation
from sklearn.learning_curve import learning_curve

print "Importing data..."
data = sp.loadmat("data.mat")
x = data['X']
y = data['y']
xtest = data['Xtest']
ytest = data['ytest']
xval = data['Xval']
yval = data['yval']
x = np.concatenate([x,xtest,xval])
y = np.concatenate([y,ytest,yval])
print "Imported %d samples with %d feature(s)..." % x.shape

plt.plot(x,y,'rx')
plt.xlabel("Change in water level")
plt.ylabel("Water flowing out")
plt.show()

x = x.tolist()
for i in x:
	for j in np.arange(2,8):
		i.append(i[0] ** j)
x = np.array(x)

pl = pipeline.Pipeline([('scaler',preprocessing.StandardScaler().fit(x)),('regr',linear_model.Ridge(alpha=0.1).fit(x,y))])
train_sizes,train_scores,test_scores = learning_curve(pl,x,y)

train_score_mean = np.mean(train_scores,axis=1)
train_score_dev = np.std(train_scores,axis=1)
test_score_mean = np.mean(test_scores,axis=1)
test_score_dev = np.std(test_scores,axis=1)

plt.clf()
plt.plot(train_sizes,train_score_mean,'o-',color='red',label='Training score')
plt.fill_between(train_sizes,train_score_mean+train_score_dev,train_score_mean-train_score_dev,alpha='0.1',color='r')
plt.plot(train_sizes,np.mean(test_scores,axis=1),'o-',color='green',label='Cross val score')
plt.fill_between(train_sizes,test_score_mean+test_score_dev,test_score_mean-test_score_dev,alpha='0.1',color='g')
plt.xlabel("Fraction of dataset used")
plt.ylabel("Score (lambda = 0.1)")
plt.legend(loc='lower right')
plt.show()
