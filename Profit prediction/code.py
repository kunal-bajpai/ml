import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model, cross_validation

print("Importing univariate data...")
data = np.genfromtxt("data.txt",delimiter=',')
x = data[:,0][:,None]
y = data[:,1]

print("Fitting classifier...")
regr = linear_model.LinearRegression()
regr.fit(x,y)
print "Coefficients are : {}".format(regr.coef_)
plt.plot(x,y,'rx',x,regr.predict(x))
plt.xlabel("Population of city in 10,000s")
plt.ylabel("Profit in $10,000s")
plt.show()
print "R^2 Score = %.4f" %( np.mean(cross_validation.cross_val_score(regr,x,y,cv=5)))
