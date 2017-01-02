import os
import pandas
import urllib.request
import numpy as np

def trainAndTest(data, testPercentage=0.1):
	'''
	Shuffles the data and separates in two datasets
	'''
	permutation = np.random.permutation(data.index)
	indice = int(testPercentage*len(permutation))
	train = data.ix[permutation[indice:]]
	test = data.ix[permutation[:indice]]
	return train, test

def download(url, fileName, saveDirectory):
	"""
	Downloads the given fileName
	"""
	if not(os.path.exists(saveDirectory + fileName)):
		response = urllib.request.urlopen(url + fileName)

		with open(saveDirectory + fileName, 'wb') as out:
		    out.write(response.read())

		print("Success")
	else :
		print("Data already downloaded")

def abalone(testPercentage = 0.1, dataset = "Data/abalone.data"):
	"""
	Reads the abalone dataset and returns train and test subdatasets
	"""
	data = pandas.read_csv(dataset, header=None)
	mapping = {'M':-1, 'F':1, 'I':0}
	data = data.replace({data.columns[0]:mapping})

	return trainAndTest(data, testPercentage)


if __name__ == '__main__':
	print("Download")
	download("https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/", "abalone.data", "Data/")
