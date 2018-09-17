
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op
from ex2_utils import plotData, plotDecisionBoundary, sigmoid, predict, mapFeature
## Machine Learning Online Class - Exercise 2: Logistic Regression

def costFunctionReg(theta, X, y, _lambda):
    m = y.size
    reg_g = _lambda/m * theta; reg_g[0] = 0
    hx = sigmoid(X @ theta)
    J = (- y @ np.log(hx) - (1 - y) @ np.log(1 - hx))/m + (reg_g @ theta)/2
    grad = X.T @ (hx - y) / m + reg_g
    return (J, grad)

## Load Data
#  The first two columns contains the X values and the third column
#  contains the label (y).

scriptdir = os.path.dirname(os.path.realpath(__file__))
data = np.loadtxt(scriptdir + '//ex2data2.txt', delimiter=',')
X = data[:,:-1]
y = data[:,-1]
plotData(X, y)

# Labels and Legend
plt.legend(['y = 1', 'y = 0'])
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.show()

# =========== Part 1: Regularized Logistic Regression ============
# Add Polynomial Features

# Note that mapFeature also adds a column of ones for us, so the
# intercept term is handled
X = mapFeature(X[:,0], X[:,1])

# Initialize fitting parameters
initial_theta = np.zeros(X.shape[1])

# Set regularization parameter lambda to 1
_lambda = 1
# Compute and display initial cost and gradient for regularized logistic
# regression
cost, grad = costFunctionReg(initial_theta, X, y, _lambda)

print(f'Cost at initial theta (zeros): {cost}')
print('Expected cost (approx): 0.693')
print('Gradient at initial theta (zeros) - first five values only:')
print(grad[0:5])
print('Expected gradients (approx) - first five values only:')
print(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115')

input('Program paused. Press enter to continue.\n')

# Compute and display cost and gradient
# with all-ones theta and lambda = 10
test_theta = np.ones(X.shape[1])
cost, grad = costFunctionReg(test_theta, X, y, 10)

print(f'Cost at test theta (with lambda = 10): {cost}')
print(f'Expected cost (approx): 3.16')
print(f'Gradient at test theta - first five values only:')
print(grad[0:5])
print('Expected gradients (approx) - first five values only:')
print(' 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922')

input('\nProgram paused. Press enter to continue.\n')

# ============= Part 2: Regularization and Accuracies =============
#  Try the following values of lambda (0, 1, 10, 100).
#  How does the decision boundary change when you vary lambda? How does
#  the training set accuracy vary?

# Initialize fitting parameters
initial_theta = np.zeros(X.shape[1])

_lambda = [0, 1, 10, 100]
expected = [87.3, 83.1, 74.6, 61.0]
for i in range(len(_lambda)):
    # Optimize
    Result = op.minimize(fun = costFunctionReg, x0 = initial_theta, args = (X, y, _lambda[i]), method = 'TNC', jac = True)
    theta = Result.x

    # Plot Boundary
    plotDecisionBoundary(theta, X, y)
    plt.title(f'lambda = {_lambda[i]}')
    # Labels and Legend
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.legend(['y = 1', 'y = 0', 'Decision boundary'])
    plt.show()

    # Compute accuracy on our training set
    p = predict(theta, X)

    print(f'Train Accuracy: {np.mean(p == y) * 100}')
    print(f'Expected accuracy (with lambda = {_lambda[i]}): {expected[i]} (approx)')
    input('\nProgram paused. Press enter to continue.\n')

