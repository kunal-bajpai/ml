import numpy as np
import csv
import math

from sklearn.feature_extraction import DictVectorizer

def getData():
	datasets = ['20_News_Group_Dataset','abc','classic','gutenberg','problem_reports','Reuters_Dataset','Sentiment_analysis','SMS_Spam_Dataset']
	datafiles = ['X_count.txt','X_tf_norm.txt','X_tf_times_idf_norm.txt']
	print ""
	for i,dataset in enumerate(datasets):
		print "\t%d. %s" % (i+1,dataset)
	dataset = datasets[int(raw_input("Please choose dataset to use : "))-1]
	print ""
	for i,datafile in enumerate(datafiles):
		print "\t%d. %s" % (i+1,datafile)
	datafile = datafiles[int(raw_input("Please choose file to use : "))-1]
	y=[]
	x=[]

	print "\nImporting data..."
	file = csv.reader(open("../datasets/"+dataset+"/preprocessed/preprocessed_data/"+datafile))
	for row in file:
		temp = row[0].split()
		y.append(int(temp[0]))
		d = dict()
		for i in temp[1:]:
			d[i.split(':')[0]] = float(i.split(':')[1])
		x.append(d)
	print "Data imported"

	print "\nConverting to sparse matrix..."
	dv = DictVectorizer(sparse=True,dtype=np.float)
	x = dv.fit_transform(x)
	print "Obtained %d samples with %d features" % x.shape

	y = np.array(y)
	trainx = x[:-math.floor(0.25*x.shape[0]),:]
	trainy = y[:-math.floor(0.25*x.shape[0])]
	testx = x[-math.floor(0.25*x.shape[0]):,:]
	testy = y[-math.floor(0.25*x.shape[0]):]
	return x,y,trainx,trainy,testx,testy
