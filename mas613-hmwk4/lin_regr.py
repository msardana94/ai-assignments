from sklearn.model_selection import KFold
from sklearn.feature_selection import mutual_info_classif
import pandas as pd
import numpy as np
import sys,csv

def GD_part1(traindata,w0,w1,alpha):
	dw0=0
	dw1=0
	for row in traindata:
		dw0 += row[1]-(w0+w1*row[0])
		dw1 += (row[1]-(w0+w1*row[0]))*row[0]
	w0 += alpha*dw0
	w1 += alpha*dw1

	return w0,w1

def GD_part2(traindata,w,alpha,m):
	# print traindata.describe()
	dw=[0 for i in range(len(w))]
	for row in traindata.itertuples(index=False):
		row = [1]+list(row)
		X=row[:-1]
		y=row[-1]
		dy = y-sum([w[i]*X[i] for i in range(len(w))])
		# print dy
		dy = [row[i]*dy for i in range(len(row))]
		dw = [alpha*sum(i)/m for i in zip(dw,dy)]
	# print dw
	w = [sum(i) for i in zip(dw,w)]

	return w


def part1(trainfile,testfile):
	w1=0
	w0=0
	alpha=0.001
	traindata = []
	i=0
	with open(trainfile) as f:
		data = csv.reader(f)
		for row in data:
			traindata.append((float(row[0]),float(row[1])))
	while True:
		oldw0 = w0
		oldw1 = w1
		w0,w1 = GD_part1(traindata,w0,w1,alpha)
		if i%50==0:
			print("iterations=%d, w0=%f, w1=%f" %(i,w0,w1))
		i+=1
		converge = pow(pow(oldw0-w0,2)+pow(oldw1-w1,2),0.5)
		if converge<0.0001:
			break

	print("\n\nFinal w0 value:%f \nFinal w1 value:%f" %(w0,w1))
	predicted = []
	actual = []
	with open(testfile) as f:
		testdata = csv.reader(f)
		for row in testdata:
			row[0] = float(row[0])
			row[1] = float(row[1])
			predicted.append(w0+w1*row[0])
			actual.append(row[1])

	mse = sum([pow(i-j,2) for i,j in zip(predicted,actual)])/len(predicted)
	print("\nMean squared error:%f\n" %mse)


def part2(datafile):
	alpha=0.1
	corr_threshold=0.1
	names = ["age","mother_edu","father_edu","travel_time","study_time","past_failure","family_rel","free_time","going_out","weekday_alcohol","weekend_alcohol","health_status","school_abs","grade"]
	with open(datafile) as f:
		fdata = pd.read_csv(f,names = names)

	# mutual_info = mutual_info_classif(fdata.as_matrix(columns=list(set(names) - set(["grade"]))),fdata['grade'])
	# # print mutual_info
	# # print np.where(mutual_info>=0.1)[0].tolist()
	# colnames = [names[i] for i in np.where(mutual_info>=0.1)[0].tolist()] + ['grade']
	fdata_norm = (fdata - fdata.mean()) / (fdata.max() - fdata.min())

	# print fdata_norm.as_matrix(columns=list(set(names) - set("grade")))
	# print type(fdata_norm['grade'])
	# print list(set(names) - set(["grade"]))

	corr_coeff = fdata_norm.corr()
	# print corr_coeff
	corr_coeff = corr_coeff[['grade']]
	colnames = list(corr_coeff[(abs(corr_coeff) >= corr_threshold).any(axis=1)].index)

	print "\nFeatures selected: "+str(colnames[:-1])+"\n"
	fdata_filtered = fdata_norm[colnames]
	# print fdata_filtered.describe()

	w = [0 for i in range(len(colnames))]
	mse=0
	kf = KFold(n_splits=5)
	for train, test in kf.split(fdata_filtered):
		traindata = fdata_filtered.iloc[train,]
		testdata = fdata_filtered.iloc[test,]
		m=len(train)
		w = [0 for i in range(len(colnames))]
		itr=0
		while True:
			oldw = w
			w = GD_part2(traindata,w,alpha,m)
			# print w
			# break
			if itr%10000==0:
				print("iterations="+str(itr)+", w="+ str(w))
			itr+=1
			converge = pow(sum([pow(oldw[i]-w[i],2) for i in range(len(w))]),0.5)
			if itr>10000 and converge<0.00001:
				break
		print("\n\nw value:"+str(w))
		predicted = []
		actual = []
		
		for row in testdata.itertuples(index=False):
			row = [1]+list(row)
			X=row[:-1]
			y=row[-1]
			predicted.append(sum([w[i]*float(X[i]) for i in range(len(w))]))
			actual.append(row[1])

		mse += sum([pow(i-j,2) for i,j in zip(predicted,actual)])/len(predicted)
	print("\nMean squared error:%f\n" %mse)

if __name__ == '__main__':
	if sys.argv[1]=="part1" and len(sys.argv)==4:
		part1(sys.argv[2],sys.argv[3])
	elif sys.argv[1]=="part2" and len(sys.argv)==3:
		part2(sys.argv[2])
	else:
		print "Invalid command. Format to execute the script:Part1--> python lin_regr.py part1 trainfile testfile ; Part2--> python lin_regr.py part2 filename"