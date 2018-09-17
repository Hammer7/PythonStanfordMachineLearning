
import os
import numpy as np
import matplotlib.pyplot as plt

## Machine Learning Online Class
#  Exercise 1: Linear regression with multiple variables

def featureNormalize(X):
    mean = np.mean(X, 0)
    sigma = np.std(X, 0)
    X_norm = (X - mean) / sigma
    return (X_norm, mean, sigma)

def computeCostMulti(X, y, theta):
    error = (X @ theta - y)
    return (error.T @ error)/ (2*y.size)


def gradientDescentMulti(X, y, theta, alpha, num_iters):
    m = y.size
    J_history = np.zeros(num_iters)
    for iter in range(num_iters):
        theta -= alpha/m * X.T @ (X @ theta - y)
        J_history[iter] = computeCostMulti(X, y, theta)
    return (theta, J_history)

def normalEqn(X,y):
    return np.linalg.pinv(X.T @ X) @ X.T @ y

# ================ Part 1: Feature Normalization ================

print('Loading data ...')
# Load Data
scriptdir = os.path.dirname(os.path.realpath(__file__))
data = np.loadtxt(scriptdir + '//ex1data2.txt', delimiter=',')
X = data[:,:-1]
y = data[:,-1]

# Print out some data points
print('First 10 examples from the dataset: X')
for i in range(10):
    print(f'x = {X[i,:]}, y = {y[i]}')
input('Program paused. Press enter to continue.\n')

# Scale features and set them to zero mean
print('Normalizing Features ...')
X, mean, sigma = featureNormalize(X)
##Add intercept term to X
X = np.c_[np.ones(y.size), X]

# ================ Part 2: Gradient Descent ================
print('Running gradient descent ...')
alpha = [0.01, 0.03, 0.10, 0.3, 1, 1.25]
for i in range(len(alpha)):
    #Choose some alpha value
    num_iters = 400

    #Init Theta and Run Gradient Descent 
    theta = np.zeros(3)
    theta, J_history = gradientDescentMulti(X, y, theta, alpha[i], num_iters)

    #Plot the convergence graph
    plt.plot(J_history, lw=2, label=f'Cost J for alpha {alpha[i]}')

    #Display gradient descent's result
    print(f'Theta computed from gradient descent with alpha {alpha[i]}')
    print(f'{theta}\n')
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.legend()
plt.show()

# Estimate the price of a 1650 sq-ft, 3 br house
house = [1650, 3]
house_norm = ((house - mean) / sigma)
price = np.append(1, house_norm) @ theta
print(f'Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n ${price}')
input('Program paused. Press enter to continue.\n')


# ================ Part 3: Normal Equations ================

print('Solving with normal equations...')

X = data[:,:-1]
y = data[:,-1]

#Add intercept term to X
X = np.c_[np.ones(y.size), X] 

# Calculate the parameters from the normal equation
theta = normalEqn(X, y)

# Display normal equation's result
print('Theta computed from the normal equations:')
print(theta)
input('Program paused. Press enter to continue.\n')


# Estimate the price of a 1650 sq-ft, 3 br house
house = [1650, 3]
price = np.append(1, house) @ theta
print(f'Predicted price of a 1650 sq-ft, 3 br house (using normal equations):\n ${price}')