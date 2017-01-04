import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import gradientDescent as gd

MAXITER = 500
LRATE = 0.01
TTIME = 10

ttime = range(0, MAXITER, TTIME)

print("TESTS : {} iterations, learning rate {}".format(MAXITER, LRATE))
print("GRADIENT DESCENT")
weight, train, test = gd.gradientDescent(maxIter = MAXITER, learningRate = LRATE, testTime = TTIME)
print("STOCHASTIC GRADIENT DESCENT")
weight, trainS, testS = gd.stochasticGradientDescent(maxIter = MAXITER, learningRate = LRATE, testTime = TTIME)
print("EVE GRADIENT DESCENT")
weight, trainE, testE = gd.eveGradientDescent(maxIter = MAXITER, learningRate = LRATE, testTime = TTIME)

plt.figure(1)

# Training error
plt.subplot(211)
plt.plot(ttime, train, color="blue", linewidth=1.0, linestyle="-", label="GD")
plt.plot(ttime, trainS, color="red", linewidth=1.0, linestyle="-", label="SGD")
plt.plot(ttime, trainE, color="green", linewidth=1.0, linestyle="-", label="Eve")
plt.xlim(0, MAXITER)
plt.xlabel('Number of iterations')
plt.ylabel('Error')
plt.title('Training error')

# log
plt.subplot(212)
plt.plot(ttime, test, color="blue", linewidth=1.0, linestyle="-", label="GD")
plt.plot(ttime, testS, color="red", linewidth=1.0, linestyle="-", label="SGD")
plt.plot(ttime, testE, color="green", linewidth=1.0, linestyle="-", label="Eve")
plt.xlim(0, MAXITER)
plt.xlabel('Number of iterations')
plt.ylabel('Error')
plt.title('Testing error')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()
