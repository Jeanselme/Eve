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

def gradientDescent(train, trainLabels, test, testLabels,
	maxIter = 10, learningRate = 0.001, regularization = 0.01, testTime = 2):
	"""
	Computes the gradient descent in order to predict the labels
	-> Binary classification by logistic regression
	"""
	lossesTest = []
	lossesTrain = []
	weight = np.zeros(len(train.columns))
	for i in range(maxIter):
		loss = 0
		grad = np.zeros(weight.shape)

		for j in range(len(train)):
			grad += logisticGrad(train.iloc[j], trainLabels.iloc[j], weight)/len(train)
			loss += logisticLoss(train.iloc[j], trainLabels.iloc[j], weight)/len(train)

		weight -= learningRate*(grad + regularization*weight)


		if (i % testTime == 0):
			print("Iteration : {} / {}".format(i+1, maxIter))
			print("\t-> Train Loss : {}".format(loss))
			lossesTrain.append(loss)
			loss = 0
			for j in range(len(test)):
				loss += logisticLoss(test.iloc[j], testLabels.iloc[j], weight)/len(test)
			lossesTest.append(loss)
			print("\t-> Test Loss : {}".format(loss))

	return weight, lossesTrain, lossesTest

def stochasticGradientDescent(train, trainLabels, test, testLabels,
	maxIter = 10, learningRate = 0.001, regularization = 0.01, testTime = 2):
	"""
	Computes the stochastic gradient descent in order to predict labels
	-> Binary classification
	"""
	lossesTest = []
	lossesTrain = []
	weight = np.zeros(len(train.columns))
	for i in range(maxIter * len(train)):
		j = rand.randint(len(train))
		grad = logisticGrad(train.iloc[j], trainLabels.iloc[j], weight)/len(train)

		weight -= learningRate*(grad + regularization*weight)

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

def adamGradientDescent(train, trainLabels, test, testLabels,
	maxIter = 10, learningRate = 0.001, regularization = 0.01, testTime = 2,
	b1 = 0.9, b2 = 0.999, b3 = 0.999, epsilon = 10**(-8), k = 0.1, K = 10):
	"""
	Computes the gradient descent in order to predict labels thanks to the eve algorithm
	-> Binary classification
	"""
	lossesTest = []
	lossesTrain = []
	weight = np.zeros(len(train.columns))

	m = np.zeros(weight.shape)
	v = np.zeros(weight.shape)
	b1t = 1
	b2t = 1

	for i in range(maxIter):
		b1t *= b1
		b2t *= b2
		loss = 0
		grad = np.zeros(weight.shape)

		for j in range(len(train)):
			grad += logisticGrad(train.iloc[j], trainLabels.iloc[j], weight)/len(train)
			loss += logisticLoss(train.iloc[j], trainLabels.iloc[j], weight)/len(train)

		m = b1*m + (1-b1)*grad
		mh = m / (1-b1t)

		v = b2*v + (1-b2)*np.multiply(grad,grad)
		vh = v/(1-b2t)

		weight -= learningRate*(np.multiply(mh,1/(np.sqrt(vh) + epsilon)) + regularization*weight)

		if (i % testTime == 0):
			print("Iteration : {} / {}".format(i+1, maxIter))
			print("\t-> Train Loss : {}".format(loss))
			lossesTrain.append(loss)
			loss = 0
			for j in range(len(test)):
				loss += logisticLoss(test.iloc[j], testLabels.iloc[j], weight)/len(test)
			lossesTest.append(loss)
			print("\t-> Test Loss : {}".format(loss))

	return weight, lossesTrain, lossesTest

def eveGradientDescent(train, trainLabels, test, testLabels,
	maxIter = 10, learningRate = 0.001, regularization = 0.01, testTime = 2,
	b1 = 0.9, b2 = 0.999, b3 = 0.999, epsilon = 10**(-8), k = 0.1, K = 10):
	"""
	Computes the gradient descent in order to predict labels thanks to the eve algorithm
	-> Binary classification
	"""
	lossesTest = []
	lossesTrain = []
	weight = np.zeros(len(train.columns))

	m = np.zeros(weight.shape)
	v = np.zeros(weight.shape)
	d = 1
	oldLoss = 0
	b1t = 1
	b2t = 1

	for i in range(maxIter):
		b1t *= b1
		b2t *= b2
		loss = 0
		grad = np.zeros(weight.shape)

		for j in range(len(train)):
			grad += logisticGrad(train.iloc[j], trainLabels.iloc[j], weight)/len(train)
			loss += logisticLoss(train.iloc[j], trainLabels.iloc[j], weight)/len(train)

		m = b1*m + (1-b1)*grad
		mh = m / (1-b1t)

		v = b2*v + (1-b2)*np.multiply(grad,grad)
		vh = v/(1-b2t)

		if (i > 0):
			if loss < oldLoss:
				delta = k + 1
				Delta = K + 1
			else:
				delta = 1/(K+1)
				Delta = 1/(k+1)
			c = min(max(delta, loss/oldLoss), Delta)
			oldLossS = oldLoss
			oldLoss = c*oldLoss
			r = abs(oldLoss - oldLossS)/(min(oldLoss,oldLossS))
			d = b3*d + (1-b3)*r
		else:
			oldLoss = loss

		weight -= learningRate*(np.multiply(mh,1/(d*np.sqrt(vh) + epsilon)) + regularization*weight)


		if (i % testTime == 0):
			print("Iteration : {} / {}".format(i+1, maxIter))
			print("\t-> Train Loss : {}".format(loss))
			lossesTrain.append(loss)
			loss = 0
			for j in range(len(test)):
				loss += logisticLoss(test.iloc[j], testLabels.iloc[j], weight)/len(test)
			lossesTest.append(loss)
			print("\t-> Test Loss : {}".format(loss))

	return weight, lossesTrain, lossesTest
