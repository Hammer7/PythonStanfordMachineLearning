import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import scipy.optimize as op

## Machine Learning Online Class - Exercise 4 Neural Network Learning
def displayData(X, width=None):
    m, n = X.shape
    if width is None:
        width = np.round(np.sqrt(n))
    width = int(width)
    height = int(n / width)
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))
    pad = 1
    display_array = -np.ones((pad + display_rows * (height + pad), pad + display_cols * (width + pad)))
    curr_ex = 0
    for y in range(display_rows):
        for x in range(display_cols):
            if curr_ex >= m:
                break
            max_val = max(abs(X[curr_ex, :]))
            xstart = pad + x *(width+pad)
            ystart = pad + y* (height+pad)
            display_array[np.ix_(range(ystart, ystart + height), range(xstart,xstart + width))] =  X[curr_ex, :].reshape((height, width)).T / max_val
            curr_ex = curr_ex + 1
        if curr_ex >= m:
	        break
    plt.imshow(display_array, cmap='gray')
    plt.show()

def predict(Theta1, Theta2, X):
    m = X.shape[0]
    X = np.c_[np.ones(m), X]
    A2 = np.c_[np.ones(m), sigmoid(X @ Theta1.T)]
    pred = np.argmax(sigmoid(A2 @ Theta2.T), axis=1) #A3
    return pred + 1

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoidGradient(z):
    sig = sigmoid(z)
    return sig * (1 - sig)

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, _lambda):
    m = y.size
    X = np.c_[np.ones(m), X]

    y_1hot = np.zeros((y.size, np.max(y)))
    y_1hot[np.arange(y.size),y-1] = 1

    t1_count = hidden_layer_size * (input_layer_size + 1)
    Theta1 = nn_params[: t1_count].reshape(hidden_layer_size, input_layer_size + 1)
    Theta2 = nn_params[t1_count : ].reshape(num_labels, -1)

    Z2 = X @ Theta1.T
    A2 = np.c_[np.ones(m), sigmoid(Z2)]
    hx = sigmoid(A2 @ Theta2.T) # A3 5000 x 10

    reg1_g = _lambda * Theta1 / m; reg1_g[:, 0] = 0
    reg2_g = _lambda * Theta2 / m; reg2_g[:, 0] = 0

    J = np.sum(- y_1hot * np.log(hx) - (1 - y_1hot) * np.log(1 - hx))/m + np.sum(reg1_g * Theta1)/2 + np.sum(reg2_g * Theta2)/2

    delta3 = hx - y_1hot # 5000 x 10
    delta2 = delta3 @ Theta2[:, 1:] * sigmoidGradient(Z2) # 5000 x 25

    Theta2_grad = (delta3.T @ A2) / m + reg2_g # 10 x 26
    Theta1_grad = (delta2.T @ X) / m + reg1_g #25 x 401

    return (J, np.concatenate([Theta1_grad.ravel(), Theta2_grad.ravel()]))

def randInitializeWeights(L_in, L_out):
    epsilon_init = 0.12
    return np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init

def debugInitializeWeights(fan_out, fan_in):
    # Initialize W using "sin", this ensures that W is always of the same
    # values and will be useful for debugging
    m = (fan_in + 1) * fan_out
    W = np.sin(range(m)).reshape(fan_out, fan_in + 1)
    return W

def computeNumericalGradient(theta, input_layer_size, hidden_layer_size, num_labels, X, y, _lambda):
    numgrad = np.zeros(theta.size)
    perturb = np.zeros(theta.size)
    e = 1e-4
    for p in range(theta.size):
        # Set perturbation vector
        perturb[p] = e
        J1, _ = nnCostFunction(theta - perturb, input_layer_size, hidden_layer_size, num_labels, X, y, _lambda)
        J2, _ = nnCostFunction(theta + perturb, input_layer_size, hidden_layer_size, num_labels, X, y, _lambda)
        # Compute Numerical Gradient
        numgrad[p] = (J2 - J1) / (2*e)
        perturb[p] = 0
    return numgrad

def checkNNGradients(_lambda=0):
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5
    # We generate some 'random' test data
    Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    Theta2 = debugInitializeWeights(num_labels, hidden_layer_size)
    # Reusing debugInitializeWeights to generate X
    X  = debugInitializeWeights(m, input_layer_size - 1)
    y  = 1 + np.mod(range(m), num_labels)

    # Unroll parameters
    nn_params = np.concatenate([Theta1.ravel(), Theta2.ravel()])

    _, grad = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, _lambda)
    numgrad = computeNumericalGradient(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, _lambda)

    # Visually examine the two gradient computations.  The two columns
    # you get should be very similar.
    for i in range(numgrad.size):
        print(f'[{numgrad[i]}, {grad[i]}]')
    print('The above two columns you get should be very similar.')
    print('(Left-Your Numerical Gradient, Right-Analytical Gradient)\n')

    # Evaluate the norm of the difference between two solutions.  
    # If you have a correct implementation, and assuming you used EPSILON = 0.0001 
    # in computeNumericalGradient, then diff below should be less than 1e-9
    diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)

    print('If your backpropagation implementation is correct, then ')
    print('the relative difference will be small (less than 1e-9). ')
    print(f'\nRelative Difference: {diff}')


# =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  You will be working with a dataset that contains handwritten digits.

# Load Training Data
print('Loading and Visualizing Data ...')

scriptdir = os.path.dirname(os.path.realpath(__file__))
data = loadmat(scriptdir + '//ex4data1.mat') # training data stored in arrays X, y
X = data['X']
y = data['y'].ravel()
num_labels = 10          # 10 labels, from 1 to 10

m, input_layer_size = X.shape #400 = 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units

# Randomly select 100 data points to display
sel = np.random.choice(m, 100, replace=False)
displayData(X[sel, :])

input('Program paused. Press enter to continue.\n')

# ================ Part 2: Loading Pameters ================
# In this part of the exercise, we load some pre-initialized 
# neural network parameters.

print('\nLoading Saved Neural Network Parameters ...')

# Load the weights into variables Theta1 and Theta2
weight = loadmat(scriptdir + '//ex4weights.mat')
Theta1 = weight['Theta1']
Theta2 = weight['Theta2']
nn_params = np.concatenate([Theta1.ravel(), Theta2.ravel()])


## ================ Part 3: Compute Cost (Feedforward) ================
#
print('\nFeedforward Using Neural Network ...')

# Weight regularization parameter (we set this to 0 here).
_lambda = 0

J, _ = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, _lambda)

print(f'Cost at parameters (loaded from ex4weights): {J}')
print('this value should be about 0.287629)')

input('\nProgram paused. Press enter to continue.\n')

## =============== Part 4: Implement Regularization ===============
print('\nChecking Cost Function (w/ Regularization) ... ')

# Weight regularization parameter (we set this to 1 here).
_lambda = 1

J, _ = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, _lambda)

print(f'Cost at parameters (loaded from ex4weights): {J}')
print('(this value should be about 0.383770)')

input('Program paused. Press enter to continue.\n')

## ================ Part 5: Sigmoid Gradient  ================

print('\nEvaluating sigmoid gradient...')

g = sigmoidGradient(np.array([-1, -0,.5, 0, 0.5, 1]))
print('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:')
print(g)
print('\n\n')

input('Program paused. Press enter to continue.\n')


# ================ Part 6: Initializing Pameters ================

print('\nInitializing Neural Network Parameters ...')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

# Unroll parameters
initial_nn_params = np.concatenate([initial_Theta1.ravel(), initial_Theta2.ravel()])
print(nn_params.shape)
print(Theta1.shape)
print(Theta2.shape)

## =============== Part 7: Implement Backpropagation ===============
print('\nChecking Backpropagation... ')

#  Check gradients by running checkNNGradients
checkNNGradients()


input('\nProgram paused. Press enter to continue.\n')

## =============== Part 8: Implement Regularization ===============
#  Once your backpropagation implementation is correct, you should now
#  continue to implement the regularization with the cost and gradient.

print('\nChecking Backpropagation (w/ Regularization) ... ')

#  Check gradients by running checkNNGradients
_lambda = 3
checkNNGradients(_lambda)

# Also output the costFunction debugging values
debug_J, _  = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, _lambda)

print(f'\n\nCost at (fixed) debugging parameters (w/ lambda = {_lambda}): {debug_J} ')
print(f'(for lambda = 3, this value should be about 0.576051)\n')

input('Program paused. Press enter to continue.\n')


## =================== Part 8: Training NN ===================
print('\nTraining Neural Network... ')
#  You should also try different values of lambda
_lambda = 1
Result = op.minimize(fun = nnCostFunction, x0 = initial_nn_params, args = (input_layer_size, hidden_layer_size, num_labels, X, y, _lambda), method = 'TNC', jac = True)
nn_params = Result.x
cost = Result.fun

# Obtain Theta1 and Theta2 back from nn_params
t1_count = hidden_layer_size * (input_layer_size + 1)
Theta1 = nn_params[: t1_count].reshape(hidden_layer_size, input_layer_size + 1)
Theta2 = nn_params[t1_count : ].reshape(num_labels, -1)
input('Program paused. Press enter to continue.\n')

## ================= Part 9: Visualize Weights =================
#  You can now "visualize" what the neural network is learning by 
#  displaying the hidden units to see what features they are capturing in 
#  the data.

print('\nVisualizing Neural Network... ')

displayData(Theta1[:, 1:])

input('\nProgram paused. Press enter to continue.\n')

displayData(Theta2[:, 1:])
input('\nProgram paused. Press enter to continue.\n')

## ================= Part 10: Implement Predict =================

pred = predict(Theta1, Theta2, X)

print(f'\nTraining Set Accuracy: {np.mean(pred == y) * 100}')

