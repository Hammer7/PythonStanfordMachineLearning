import os
import numpy as np
from scipy.io import loadmat
import scipy.optimize as op
from ex3_utils import displayData, sigmoid

## Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all

def lrCostFunction(theta, X, y, _lambda):
    m = y.size
    reg = _lambda/m * theta; reg[0] = 0
    hx = sigmoid(X @ theta)
    J = (- y @ np.log(hx) - (1 - y) @ np.log(1 - hx))/m + (reg @ reg)/2
    grad = X.T @ (hx - y) / m + reg
    return (J, grad)

def oneVsAll(X, y, num_labels, _lambda):
    X = np.c_[np.ones(X.shape[0]), X]
    theta = np.zeros((num_labels, X.shape[1]))
    for i in range(1, num_labels+1):
        label = (y == i).astype(int) #y is loaded from mat file and values from 1 to 10
        Result = op.minimize(fun = lrCostFunction, x0 = np.zeros(X.shape[1]), args = (X, label, _lambda), method = 'TNC', jac = True)
        theta[i-1:] = Result.x
    return theta

def predictOneVsAll(all_theta, X):
    X = np.c_[np.ones(X.shape[0]), X]
    pred = np.argmax(sigmoid(X @ all_theta.T), axis=1)
    return pred + 1
# =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  You will be working with a dataset that contains handwritten digits.

# Load Training Data
print('Loading and Visualizing Data ...')
scriptdir = os.path.dirname(os.path.realpath(__file__))
data = loadmat(scriptdir + '//ex3data1.mat') # training data stored in arrays X, y
X = data['X']
y = data['y'].ravel()
num_labels = 10          # 10 labels, from 1 to 10
                         # note that we have mapped "0" to label 10
m, input_layer_size = X.shape # 5000 , 400 = 20x20 Input Images of Digits

# Randomly select 100 data points to display
sel = np.random.choice(m, 100, replace=False)
displayData(X[sel, :])

input('Program paused. Press enter to continue.\n')

# ============ Part 2a: Vectorize Logistic Regression ============

# Test case for lrCostFunction
print('\nTesting lrCostFunction() with regularization')

theta_t = np.array([-2, -1, 1, 2])
X_t = np.c_[np.ones(5), np.arange(1, 16).reshape((5,3), order='F')/10]
y_t = np.array([1, 0, 1, 0, 1])
lambda_t = 3
J, grad = lrCostFunction(theta_t, X_t, y_t, lambda_t)

print(f'\nCost: {J}')
print('Expected cost: 2.534819')
print('Gradients:\n')
print(grad)
print('Expected gradients:')
print(' 0.146561\n -0.548558\n 0.724722\n 1.398003')
input('Program paused. Press enter to continue.\n')

# ============ Part 2b: One-vs-All Training ============
print('\nTraining One-vs-All Logistic Regression...')
_lambda = 0.1
all_theta = oneVsAll(X, y, num_labels, _lambda)
input('Program paused. Press enter to continue.\n')

# ================ Part 3: Predict for One-Vs-All ================

pred = predictOneVsAll(all_theta, X)

print(f'Training Set Accuracy: {np.mean(pred == y) * 100}\n')
