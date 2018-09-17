import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from svm import SVM

#  Exercise 6 | Support Vector Machines

def linearKernel(x1, x2):
    return x1.T @ x2

def gaussianKernel(x1, x2, sigma):
    diff = x1 - x2
    return np.exp(-(diff.T @ diff)/(2*sigma*sigma))

def plotData(x, y):
    pos = np.where(y==1)
    neg = np.where(y==0)
    plt.plot(x[pos[0],0], x[pos[0],1], 'b+', lw = 1.5, ms = 4)
    plt.plot(x[neg[0],0], x[neg[0],1], 'ko', mfc = 'y', ms = 4)


def visualizeBoundaryLinear(X, y, model):
    w = model.w
    b = model.b
    xp = np.linspace(min(X[:,0]), max(X[:,0]), 100)
    yp = - (w[0] * xp + b)/w[1]
    plotData(X, y)
    plt.plot(xp, yp, '-b')

def visualizeBoundary(X, y, model):
    #VISUALIZEBOUNDARY plots a non-linear decision boundary learned by the SVM

    # Plot the training data on top of the boundary
    plotData(X, y)

    # Make classification predictions over a grid of values
    x1plot = np.linspace(min(X[:,0]), max(X[:,0]), 100)
    x2plot = np.linspace(min(X[:,1]), max(X[:,1]), 100)
    X1, X2 = np.meshgrid(x1plot, x2plot)
    vals = np.zeros(X1.shape)
    for i in range(X1.shape[1]):
        this_X = np.c_[X1[:, i], X2[:, i]]
        vals[:, i] = model.svmPredict(this_X)

    # Plot the SVM boundary
    plt.contour(X1, X2, vals, [0.5])

def dataset3Params(X, y, Xval, yval):
    Cs = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])
    sigmas = Cs
    model = SVM()

    costs = np.zeros((Cs.size, sigmas.size))
    for i in range(Cs.size):
        for j in range(sigmas.size):
            sigma = sigmas[j]
            #need to define gaussianKernelLambda lambda here again to capture sigma
            gaussianKernelLambda = lambda x1, x2: (gaussianKernel(x1, x2, sigma))
            gaussianKernelLambda.__name__ = 'gaussianKernel'

            model.svmTrain(X, y.astype(float), Cs[i], gaussianKernelLambda)
            predictions = model.svmPredict(Xval)
            costs[i, j] = np.mean(predictions != yval)
    i, j = np.unravel_index(np.argmin(costs), costs.shape)
    return (Cs[i], sigmas[j])

## =============== Part 1: Loading and Visualizing Data ================
print('Loading and Visualizing Data ...')

# Load from ex6data1: 
# You will have X, y in your environment
scriptdir = os.path.dirname(os.path.realpath(__file__))
data = loadmat(scriptdir + '//ex6data1.mat')
X = data['X']
y = data['y'].ravel()
# Plot training data
plotData(X, y)
plt.show()

input('Program paused. Press enter to continue.')


## ==================== Part 2: Training Linear SVM ====================
#  The following code will train a linear SVM on the dataset and plot the
#  decision boundary learned.

print('\nTraining Linear SVM ...')

# You should try to change the C value below and see how the decision
# boundary varies (e.g., try C = 1000)
C = 1
model = SVM()
#we passed y.astype(float), firstly to be sure y -1 can be assigned to singed int. 
#secondly we want to preserve original y to 0 and 1
model.svmTrain(X, y.astype(float), C, linearKernel, 1e-3, 20) 

visualizeBoundaryLinear(X, y, model)
plt.show()
input('Program paused. Press enter to continue.\n')

## =============== Part 3: Implementing Gaussian Kernel ===============

print('\nEvaluating the Gaussian Kernel ...')

x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2

#use lambda to be able to pass sigma
gaussianKernelLambda = lambda x1, x2: (gaussianKernel(x1, x2, sigma))
gaussianKernelLambda.__name__ = 'gaussianKernel'


sim = gaussianKernelLambda(x1, x2)

print(f'Gaussian Kernel between x1 = [1c 2, 1], x2 = [0, 4, -1], sigma = {sigma} :')
print(f'\t{sim}\n(for sigma = 2, this value should be about 0.324652)\n')

input('Program paused. Press enter to continue.\n')


## =============== Part 4: Visualizing Dataset 2 ================

print('Loading and Visualizing Data ...')

# Load from ex6data2: 
# You will have X, y in your environment
data = loadmat(scriptdir + '//ex6data2.mat')
X = data['X']
y = data['y'].ravel()

# Plot training data
plotData(X, y)
plt.show()

input('Program paused. Press enter to continue.\n')

## ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========

print('\nTraining SVM with RBF Kernel (this may take 1 to 2 minutes) ...')

# SVM Parameters
C = 1; sigma = 0.1

# We set the tolerance and max_passes lower here so that the code will run
# faster. However, in practi ce, you will want to run the training to
# convergence.
model.svmTrain(X, y.astype(float), C, gaussianKernelLambda)
visualizeBoundary(X, y, model)
plt.show()
input('Program paused. Press enter to continue.')


## =============== Part 6: Visualizing Dataset 3 ================

print('Loading and Visualizing Data ...')

# Load from ex6data3: 
# You will have X, y in your environment
data = loadmat(scriptdir + '//ex6data3.mat')

X = data['X']
y = data['y'].ravel()
Xval = data['Xval']
yval = data['yval'].ravel()

# Plot training data
plotData(X, y)
plt.show()

input('Program paused. Press enter to continue.\n')

## ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========

#  This is a different dataset that you can use to experiment with. Try
#  different values of C and sigma here. 


# Try different SVM Parameters here
C, sigma = dataset3Params(X, y, Xval, yval)
# Train the SVM
model.svmTrain(X, y.astype(float), C, gaussianKernelLambda)
visualizeBoundary(X, y, model)
plt.show()
input('Program paused. Press enter to continue.\n')