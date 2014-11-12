import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model, preprocessing, cross_validation

print("Importing bivariate data...")
data = np.genfromtxt("ex1data2.txt",delimiter=',')
x = data[:,:2]
y = data[:,2]
print("Fitting classifier...")
regr = linear_model.LinearRegression()
scaled_x = preprocessing.scale(x)
regr.fit(scaled_x,y)
print "Coefficients are : {}".format(regr.coef_)
plt.plot(x[:,0],x[:,1],'rx')
plt.xlabel("Size in square feet")
plt.ylabel("No. of bedrooms")
plt.show()
print "R^2 Score = %.4f" %( np.mean(cross_validation.cross_val_score(regr,scaled_x,y,cv=5)))
