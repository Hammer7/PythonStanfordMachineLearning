import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
from ex2_utils import plotData, plotDecisionBoundary, sigmoid, predict

## Machine Learning Online Class - Exercise 2: Logistic Regression

def costFunction(initial_theta, X, y):
    m = y.size
    hx = sigmoid(X @ initial_theta)
    J = (- y @ np.log(hx) - (1 - y) @ np.log(1 - hx))/m
    grad = X.T @ (hx - y) / m
    return (J, grad)

# Load Data
#  The first two columns contains the exam scores and the third column
#  contains the label.
scriptdir = os.path.dirname(os.path.realpath(__file__))
data = np.loadtxt(scriptdir + '//ex2data1.txt', delimiter=',')
X = data[:,:-1]
y = data[:,-1]


# ==================== Part 1: Plotting ====================
#  We start the exercise by first plotting the data to understand the 
#  the problem we are working with.

print('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.')

plotData(X, y)
# Labels and Legend
plt.legend(['Admitted', 'Not Admitted'])
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.show()
input('\nProgram paused. Press enter to continue.\n')

# ============ Part 2: Compute Cost and Gradient ============
#  Setup the data matrix appropriately, and add ones for the intercept term
m, n = X.shape
# Add intercept term to x and X_test
X = np.c_[np.ones(m), X]

# Initialize fitting parameters
initial_theta = np.zeros(n + 1)

# Compute and display initial cost and gradient
cost, grad = costFunction(initial_theta, X, y)

print(f'Cost at initial theta (zeros): {cost}')
print('Expected cost (approx): 0.693')
print('Gradient at initial theta (zeros):')
print(grad)
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n')

# Compute and display cost and gradient with non-zero theta
test_theta = [-24, 0.2, 0.2]
cost, grad = costFunction(test_theta, X, y)

print(f'Cost at test theta: {cost}')
print(f'Expected cost (approx): 0.218')
print(f'Gradient at test theta:')
print(grad)
print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n')

input('Program paused. Press enter to continue.')


# ============= Part 3: Optimizing using fminunc  =============
Result = op.minimize(fun = costFunction, x0 = initial_theta, args = (X, y), method = 'TNC', jac = True)
cost = Result.fun
theta = Result.x
# Print theta to screen
print(f'Cost at theta found by minimize: {cost}')
print('Expected cost (approx): 0.203')
print(f'theta: {theta}')
print('Expected theta (approx):')
print(' -25.161\n 0.206\n 0.201\n')

# Plot Boundary
plotDecisionBoundary(theta, X, y)
# Labels and Legend

plt.legend(['Admitted', 'Not Admitted', 'Decision Boundary'])
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.show()


input('Program paused. Press enter to continue.\n')

# ============== Part 4: Predict and Accuracies ==============
#  Predict probability for a student with score 45 on exam 1 
#  and score 85 on exam 2 

prob = sigmoid([1, 45, 85] @ theta)
print(f'For a student with scores 45 and 85, we predict an admission probability of {prob}')
print('Expected value: 0.775 +/- 0.002\n')

# Compute accuracy on our training set
p = predict(theta, X)

print(f'Train Accuracy: {np.mean((p == y)) * 100}')
print('Expected accuracy (approx): 89.0\n')


