
import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import scipy.optimize as op

## Machine Learning Online Class
#  Exercise 5 | Regularized Linear Regression and Bias-Variance
def linearRegCostFunction(theta, X, y, _lambda):
    error = (X @ theta - y)
    reg_g = _lambda/m * theta; reg_g[0] = 0
    J = (error.T @ error)/ (2*y.size) + (reg_g @ theta)/2
    grad = (X.T @ error)/m + reg_g
    return (J, grad)

def trainLinearReg(X, y, _lambda):
    initial_theta = np.zeros(X.shape[1])
    res = op.minimize(fun = linearRegCostFunction, x0 = initial_theta, args = (X, y, _lambda), method = 'TNC', jac = True)
    return res.x

def learningCurve(X, y, Xval, yval, _lambda):
    m = y.size
    Jtrain = np.zeros(m)
    Jval = np.zeros(m)
    for i in range(m):
        theta = trainLinearReg(X[0:i,:], y[0:i], _lambda)
        Jtrain[i], _ = linearRegCostFunction(theta, X[0:i,:], y[0:i], 0)
        Jval[i], _ = linearRegCostFunction(theta, Xval, yval, 0)
    return (Jtrain, Jval)

def polyFeatures(X, p):
    X_poly = np.ones((X.shape[0], p))
    for i in range(p):
        X_poly[:,i] = np.power(X[:,0], i+1)
    return X_poly

def featureNormalize(X):
    mean = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    print(mean)
    print(sigma)
    X_norm = (X - mean) / sigma
    return (X_norm, mean, sigma)

def plotFit(min_x, max_x, mu, sigma, theta, p):
    #We plot a range slightly bigger than the min and max values to get
    #an idea of how the fit will vary outside the range of the data points
    x = np.arange(min_x -15 , max_x + 25, 0.05)
    # Map the X values 
    X_poly = polyFeatures(x[:, np.newaxis], p)
    X_poly = (X_poly - mu) / sigma
    X_poly = np.c_[np.ones(x.shape[0]), X_poly] # Add ones
    # Plot
    plt.plot(x, X_poly @ theta, '--', lw = 2)

def validationCurve(X, y, Xval, yval):
    lambda_vec = np.array([0 ,0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])
    error_train = np.zeros(lambda_vec.size)
    error_val = np.zeros(lambda_vec.size)
    for i in range(lambda_vec.size):
        theta = trainLinearReg(X, y, lambda_vec[i])
        error_train[i], _ = linearRegCostFunction(theta, X, y, 0)
        error_val[i], _ = linearRegCostFunction(theta, Xval, yval, 0)
    return (lambda_vec, error_train, error_val)

## =========== Part 1: Loading and Visualizing Data =============

# Load Training Data
print('Loading and Visualizing Data ...')

# Load from ex5data1: 
# You will have X, y, Xval, yval, Xtest, ytest in your environment
scriptdir = os.path.dirname(os.path.realpath(__file__))
data = loadmat(scriptdir + '//ex5data1.mat') 
X = data['X']
y = data['y'].ravel()
Xval = data['Xval']
yval = data['yval'].ravel()
Xtest = data['Xtest']
ytest = data['ytest'].ravel()
# m = Number of examples
m = y.size

# Plot training data
plt.plot(X, y, 'rx', ms = 4, lw = 1.5)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.show()
input('Program paused. Press enter to continue.\n')

## =========== Part 2: Regularized Linear Regression Cost =============

theta = np.array([1, 1])
J, _ = linearRegCostFunction(theta, np.c_[np.ones(m), X], y, 1)

print(f'Cost at theta = [1, 1]: {J} ')
print('(this value should be about 303.993192)\n')

input('Program paused. Press enter to continue.\n')

## =========== Part 3: Regularized Linear Regression Gradient =============


theta = np.array([1, 1])
_, grad = linearRegCostFunction(theta, np.c_[np.ones(m), X], y, 1)

print(f'Gradient at theta = [1 ; 1]:  [{grad[0]}, {grad[1]}] ')
print('(this value should be about [-15.303016; 598.250744])')

input('Program paused. Press enter to continue.\n')

## =========== Part 4: Train Linear Regression ============= 
#  Write Up Note: The data is non-linear, so this will not give a great fit.

#  Train linear regression with lambda = 0
_lambda = 0
theta = trainLinearReg(np.c_[np.ones(m), X], y, _lambda)

#  Plot fit over the data
plt.plot(X, y, 'rx', ms = 4, lw = 1.5)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.plot(X, np.c_[np.ones(m), X] @  theta, '--', lw = 2)
plt.show()
input('Program paused. Press enter to continue.\n')

## =========== Part 5: Learning Curve for Linear Regression =============
#  Write Up Note: Since the model is underfitting the data, we expect to
#                 see a graph with "high bias" -- Figure 3 in ex5.pdf 

_lambda = 0
error_train, error_val = learningCurve(np.c_[np.ones(m), X], y, np.c_[np.ones(yval.size), Xval], yval, _lambda)

plt.plot(range(m), error_train)
plt.plot(range(m), error_val)
plt.title('Learning curve for linear regression')
plt.legend(['Train', 'Cross Validation'])
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.axis([0, 13, 0, 150])
plt.show()
print('# Training Examples\tTrain Error\tCross Validation Error')
for i in range(m):
    print(f'  \t{i}\t\t{error_train[i]}\t{error_val[i]}')

input('Program paused. Press enter to continue.\n')

## =========== Part 6: Feature Mapping for Polynomial Regression =============

p = 8

# Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X, p)
X_poly, mu, sigma = featureNormalize(X_poly)  # Normalize
X_poly = np.c_[np.ones(m), X_poly]                  # Add Ones

# Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest, p)
X_poly_test = (X_poly_test - mu) / sigma
X_poly_test = np.c_[np.ones(X_poly_test.shape[0]), X_poly_test] #Add Ones

# Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p)
X_poly_val = (X_poly_val - mu) / sigma
X_poly_val = np.c_[np.ones(X_poly_val.shape[0]), X_poly_val] #Add Ones

print('Normalized Training Example 1:')
print(X_poly[0])

input('\nProgram paused. Press enter to continue.\n')

## =========== Part 7: Learning Curve for Polynomial Regression =============
# You should try running the code with different values of
#  lambda to see how the fit and learning curve change.

_lambda = 0
theta = trainLinearReg(X_poly, y, _lambda)

# Plot training data and fit
plt.plot(X, y, 'rx', ms = 4, lw = 1.5)
plotFit(min(X), max(X), mu, sigma, theta, p)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.title(f'Polynomial Regression Fit (lambda = {_lambda})')
plt.show()

error_train, error_val = learningCurve(X_poly, y, X_poly_val, yval, _lambda)
plt.plot(range(m), error_train)
plt.plot(range(m), error_val)

plt.title(f'Polynomial Regression Learning Curve (lambda = {_lambda})')
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.axis([0, 13, 0, 100])
plt.legend(['Train', 'Cross Validation'])
plt.show()

print(f'Polynomial Regression (lambda = {_lambda})\n')
print('# Training Examples\tTrain Error\tCross Validation Error')
for i in range(m):
    print(f'  \t{i}\t\t{error_train[i]}\t{error_val[i]}')


input('Program paused. Press enter to continue.\n')


## =========== Part 8: Validation for Selecting Lambda =============

lambda_vec, error_train, error_val = validationCurve(X_poly, y, X_poly_val, yval)

plt.plot(lambda_vec, error_train, lambda_vec, error_val)
plt.legend(['Train', 'Cross Validation'])
plt.xlabel('lambda')
plt.ylabel('Error')
plt.show()

print('lambda\t\tTrain Error\tValidation Error')
for i in range(lambda_vec.size):
	print(f' {lambda_vec[i]}\t{error_train[i]}\t{error_val[i]}')

input('Program paused. Press enter to continue.\n')

index = np.argmin(error_val)
theta = trainLinearReg(X_poly, y, lambda_vec[index])
testerror, _ = linearRegCostFunction(theta, X_poly_test, ytest, 0)

print(f'Test error for lambda {lambda_vec[index]} = {testerror}.\n')

input('Program paused. Press enter to continue.\n')