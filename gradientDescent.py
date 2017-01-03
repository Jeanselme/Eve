import dataExtraction
import numpy as np
import numpy.random as rand

def logisticLoss(features, label, weight):
	"""
	Computes the logistic loss
	"""
	return np.log(1 + np.exp(-label*features.dot(weight)))

def logisticGrad(features, label, weight):
	"""
	Computes the gradient of the logistic loss
	"""
	denum = 1 +  np.exp(label*features.dot(weight))
	return np.multiply(features,-label/denum)

def gradientDescent(data = dataExtraction.Data, maxIter = 10, learningRate = 0.001, testPercentage = 0.1, binar = 9, testTime = 2):
	"""
	Computes the gradient descent in order to predict the last colums of the data
	by a simple linear regression
	-> Binary classification : 9 vs all others
	"""
	train, trainLabels, test, testLabels = dataExtraction.abalone(testPercentage, data)

	trainLabels = dataExtraction.binarization(trainLabels, binar)
	testLabels = dataExtraction.binarization(testLabels, binar)

	lossesTest = []
	lossesTrain = []
	weight = np.zeros(len(train.columns))
	for i in range(maxIter):
		print("Iteration : {} / {}".format(i+1, maxIter))
		loss = 0
		grad = np.zeros(weight.shape)

		for j in range(len(train)):
			grad += logisticGrad(train.iloc[j], trainLabels.iloc[j], weight)
			loss += logisticLoss(train.iloc[j], trainLabels.iloc[j], weight)/len(train)

		weight -= learningRate*grad
		print("\t-> Train Loss : {}".format(loss))

		if (i % testTime == 0):
			lossesTrain.append(loss)
			loss = 0
			for j in range(len(test)):
				loss += logisticLoss(test.iloc[j], testLabels.iloc[j], weight)/len(test)
			lossesTest.append(loss)
			print("\t-> Test Loss : {}".format(loss))

	return weight, lossesTrain, lossesTest

def stochasticGradientDescent(data = dataExtraction.Data, maxIter = 10, learningRate = 0.001, testPercentage = 0.1, binar = 9, testTime = 2):
	"""
	Computes the stochastic gradient descent in order to predict the last colums of the data
	by a simple linear regression
	-> Binary classification : 9 vs all others
	"""
	train, trainLabels, test, testLabels = dataExtraction.abalone(testPercentage, data)

	trainLabels = dataExtraction.binarization(trainLabels, binar)
	testLabels = dataExtraction.binarization(testLabels, binar)

	lossesTest = []
	lossesTrain = []
	weight = np.zeros(len(train.columns))
	for i in range(maxIter * len(train)):
		j = rand.randint(len(test))
		grad = logisticGrad(train.iloc[j], trainLabels.iloc[j], weight)

		weight -= learningRate*grad

		if (i % (testTime*len(train)) == 0):
			print("Iteration : {} / {}".format(i+1, maxIter*len(train)))
			loss = 0
			for j in range(len(train)):
				loss += logisticLoss(train.iloc[j], trainLabels.iloc[j], weight)/len(train)
			lossesTrain.append(loss)
			print("\t-> Train Loss : {}".format(loss))

			loss = 0
			for j in range(len(test)):
				loss += logisticLoss(test.iloc[j], testLabels.iloc[j], weight)/len(test)
			lossesTest.append(loss)
			print("\t-> Test Loss : {}".format(loss))

	return weight, lossesTrain, lossesTest

if __name__ == '__main__':
	print(stochasticGradientDescent())
