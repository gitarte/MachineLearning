import math
import numpy as np
import extractFeatures as ef
from sklearn import svm

np.set_printoptions(threshold=np.nan)

#	Extracting dataset
dataset   = np.genfromtxt('dataset.csv', delimiter=',')	
(m,n)     = dataset.shape			# m: number of examples, n-1: number of features
train_end = int(math.ceil(0.7*m))	# computing a teaching to testing ratio
trainSet  = dataset[0:train_end,:]	# 70% of a dataset will be a teaching data
testSet   = dataset[train_end:,:]	# 30% of a dataset will be a testing  data
X         = trainSet[:,0:n-1]		# extracting values of features from teaching data
y         = trainSet[:,n-1]			# extracting values of classes  from teaching data (0: is plain text, 1: is hash)
Xtest     = trainSet[:,0:n-1]		# extracting values of features from testing  data
ytest     = trainSet[:,n-1]			# extracting values of classes  from testing  data (0: is plain text, 1: is hash)

#	PERFORMING MACHINE LEARNING
clf = svm.SVC()				# Support Vector Machine Classifier
clf.fit(X, y)				# teaching 
result = clf.predict(Xtest)	# prediction on test examples

#	EVALUATION OF MACHINE LEARNING
compare    = (result == ytest)						# True means that prediction was correct
possitives = sum(1 for i in compare if i == True)	# counting correct predictions 
total      = len(compare)							# 
accuracy   = possitives/total*100					# the accuracy
print('Accuracy: {0}%'.format(accuracy))

#	MANUAL EVALUATION
manual_test = (
	'IYlkjHJKglh',
	'PZU',
	'PZU S.A.',
	'JanKowalski',
	'Jan Kowalski',
	'JanKo Walski',
	'LjwNoFOMYAivbZBoP',
	'Bg.pEGQBzXnJMTvbZB',
	'RwvqLUvaJMbIvbBoP',
	'QGjsqvbZBoPervJjq',
	'ovqd0bZB oPovbZBoP',
	'yvbZB2oPńvbZBoPG',
	'vbZBoP333IABiJzkf',
	'JzHpnTvbZBoPi3oJI',
	'xvehllzvbz555bpwz',
	'FZXHEVYLHJLGCCNPN',
	'dupa',
	'DUPA',
	'Dupa',
	'DuPa'
)
for example in manual_test:
	prediction = clf.predict(ef.extractFeatures(example).reshape(1, -1))
	if prediction[0]==1:
		print('Example {0}\t\tis considered a hash'.format(example))
	else:
		print('Example {0}\t\tis considered a plain text'.format(example))
