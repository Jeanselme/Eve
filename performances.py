import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import dataExtraction as de
import gradientDescent as gd

MAXITER = 500
LRATE = 0.01
TTIME = 10

ttime = range(0, MAXITER, TTIME)

train, trainLabels, test, testLabels = de.ionosphere(0.1)

print("TESTS : {} iterations, learning rate {}".format(MAXITER, LRATE))
print("GRADIENT DESCENT")
weight, trainG, testG = gd.gradientDescent(train, trainLabels, test, testLabels, maxIter = MAXITER, learningRate = LRATE, testTime = TTIME)
print("STOCHASTIC GRADIENT DESCENT")
weight, trainS, testS = gd.stochasticGradientDescent(train, trainLabels, test, testLabels, maxIter = MAXITER, learningRate = LRATE, testTime = TTIME)
print("EVE GRADIENT DESCENT")
weight, trainA, testA = gd.adamGradientDescent(train, trainLabels, test, testLabels, maxIter = MAXITER, learningRate = LRATE, testTime = TTIME)
print("EVE GRADIENT DESCENT")
weight, trainE, testE = gd.eveGradientDescent(train, trainLabels, test, testLabels, maxIter = MAXITER, learningRate = LRATE, testTime = TTIME)

plt.figure(1)

# Training error
plt.subplot(211)
plt.plot(ttime, trainG, color="blue", linewidth=1.0, linestyle="-", label="GD")
plt.plot(ttime, trainS, color="red", linewidth=1.0, linestyle="-", label="SGD")
plt.plot(ttime, trainA, color="black", linewidth=1.0, linestyle="-", label="Adam")
plt.plot(ttime, trainE, color="green", linewidth=1.0, linestyle="-", label="Eve")
plt.xlim(0, MAXITER)
plt.xlabel('Number of iterations')
plt.ylabel('Error')
plt.title('Training error')

# log
plt.subplot(212)
plt.plot(ttime, testG, color="blue", linewidth=1.0, linestyle="-", label="GD")
plt.plot(ttime, testS, color="red", linewidth=1.0, linestyle="-", label="SGD")
plt.plot(ttime, testA, color="black", linewidth=1.0, linestyle="-", label="Adam")
plt.plot(ttime, testE, color="green", linewidth=1.0, linestyle="-", label="Eve")
plt.xlim(0, MAXITER)
plt.xlabel('Number of iterations')
plt.ylabel('Error')
plt.title('Testing error')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()
