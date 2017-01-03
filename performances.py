import matplotlib.pyplot as plt
import gradientDescent as gd

MAXITER = 20
LRATE = 0.001
TTIME = 2

ttime = range(0, MAXITER, TTIME)

print("TESTS : {} iterations, learning rate {}".format(MAXITER, LRATE))
print("GRADIENT DESCENT")
weight, train, test = gd.gradientDescent(maxIter = MAXITER, learningRate = LRATE, testTime = TTIME)
print("STOCHASTIC GRADIENT DESCENT")
weight, trainS, testS = gd.stochasticGradientDescent(maxIter = MAXITER, learningRate = LRATE, testTime = TTIME)

plt.figure()
plt.plot(ttime, train, color="blue", linewidth=1.0, linestyle="-")
plt.plot(ttime, trainS, color="red", linewidth=1.0, linestyle="-")
plt.xlim(0, MAXITER)
plt.xlabel('Number of iterations')
plt.ylabel('Error')
plt.show()
