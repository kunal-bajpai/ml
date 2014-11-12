import numpy as np
import csv

from sklearn import ensemble,grid_search

print "Importing and preprocessing data..."
data = csv.reader(open("train.csv"))
header = data.next()
x=[]
y=[]
for row in data:
	temp=[]
	y.append(row[1])
	temp.append(row[2])
	temp.append(1) if row[4]=='female' else temp.append(0)
	temp.append(0) if row[5]=='' else temp.append(row[5])
	temp.append(row[6])
	temp.append(row[7])
	temp.append(row[9]) if row[9]!='' else temp.append(0)
	temp.append(1) if row[11]=='S' else temp.append(0)
	temp.append(1) if row[11]=='Q' else temp.append(0)
	temp.append(1) if row[11]=='C' else temp.append(0)
	x.append(temp)
x = np.array(x).astype(np.float)
x[:,2][x[:,2]==0] = np.median(x[:,2]!=0)
x[:,5][x[:,5]==0] = np.median(x[:,5]!=0)
print "Obtained %d samples with %d features" % x.shape

clf = grid_search.GridSearchCV(ensemble.RandomForestClassifier(),scoring='accuracy',cv=5,param_grid={'criterion':['entropy'],'max_features':[5],'min_samples_split':[3],'min_samples_leaf':[2],'n_estimators':[100]})
clf.fit(x,y)
print clf.best_estimator_
print clf.best_score_
