import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

## Machine Learning Online Class - Exercise 1: Linear Regression

def warmUpExercise():
    return np.eye(5)

def plotData(x, y):
    plt.plot(x, y, 'rx', label='Training data')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.title('Scatter plot of training data')
    plt.axis([4, 25, -5, 24])

def computeCost(X, y, theta):
    error = (X @ theta - y)
    return (error.T @ error)/ (2*y.size)

def gradientDescent(X, y, theta, alpha, iterations):
    m = y.size
    for _ in range(iterations):
        theta -= alpha/m * X.T @ (X @ theta - y)
    return theta

# ==================== Part 1: Basic Function ====================
print('Running warmUpExercise ... ')
print('5x5 Identity Matrix:')
print(warmUpExercise())
input('Program paused. Press enter to continue.\n')

# ======================= Part 2: Plotting =======================
# x refers to the population size in 10,000s
# y refers to the profit in $10,000s

scriptdir = os.path.dirname(os.path.realpath(__file__))
print(scriptdir)
data = np.loadtxt(scriptdir + '//ex1data1.txt', delimiter=',')
X = data[:,:-1]
y = data[:,-1]
m = y.size #number of training examples

print('Plotting Data ...')
#Plot Data
plotData(X, y)
plt.show()

# =================== Part 3: Cost and Gradient descent ===================
X = np.c_[np.ones(m), X] # #Add a column of ones to X (interception data)
theta = np.zeros(2) #initialize fitting parameters

# Some gradient descent settings
iterations = 1500
alpha = 0.01

print('\nTesting the cost function ...')
# compute and display initial cost
J = computeCost(X, y, theta)
print(f'With theta = [0 ; 0], Cost computed = {J}')
print('Expected cost value (approx) 32.07')


# further testing of the cost function
J = computeCost(X, y, [-1 , 2])
print(f'\nWith theta = [-1 ; 2], Cost computed = {J}')
print(f'Expected cost value (approx) 54.24')
input('Program paused. Press enter to continue.\n')


print('Running Gradient Descent ...')
# run gradient descent
theta = gradientDescent(X, y, theta, alpha, iterations)

# print theta to screen
print('Theta found by gradient descent:')
print(theta)
print('Expected theta values (approx)')
print(' -3.6303,  1.1664\n')

# Plot the linear fit
plotData(X[:,1], y)
plt.plot(X[:,1], X @ theta, '-', label='Linear regression')
plt.legend()
plt.show()

# Predict values for population sizes of 35,000 and 70,000
predict1 = [1, 3.5]  @ theta
print(f'For population = 35,000, we predict a profit of {predict1*10000}')
predict2 = [1, 7] @ theta
print(f'For population = 70,000, we predict a profit of {predict2*10000}')
input('Program paused. Press enter to continue.\n')


# ============= Part 4: Visualizing J(theta_0, theta_1) =============
print('Visualizing J(theta_0, theta_1) ...')

# Grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

# initialize J_vals to a matrix of 0's
J_vals = np.zeros((theta0_vals.size, theta1_vals.size))

# Fill out J_vals
# Because of the way Surface plot and meshgrid work, we need to swap
# j,i order in J_vals, or else the axes will be flipped
for i in range(theta0_vals.size):
    for j in range(theta1_vals.size):
	    J_vals[j,i] = computeCost(X, y, [theta0_vals[i], theta1_vals[j]])

fig = plt.figure()
ax = fig.gca(projection='3d')
# Surface plot
xx, yy = np.meshgrid(theta0_vals, theta1_vals)
surf = ax.plot_surface(xx, yy, J_vals, antialiased=False)
#LaTex rendering with matplotlib is very sow on MacOS
#ax.set_xlabel(r'$\theta_0$')
#ax.set_ylabel(r'$\theta_1$')
ax.set_xlabel('Theta_0')
ax.set_ylabel('Theta_1')
plt.show()

# Contour plot
# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
plt.contour(xx, yy, J_vals, np.logspace(-2, 3, 20))
#LaTex rendering with matplotlib is very sow on MacOS
#plt.xlabel(r'$\theta_0$')
#plt.ylabel(r'$\theta_1$')
plt.xlabel('Theta_0')
plt.ylabel('Theta_1')

plt.plot(theta[0], theta[1], 'rx', ms=10, lw=2)
plt.show()
