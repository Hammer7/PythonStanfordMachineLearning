import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.misc import imread

#  Exercise 7 | Principle Component Analysis and K-Means Clustering

def findClosestCentroids(X, centroids):
    #(||X-Y||)^2 = ||X||^2 + ||Y||^2 - 2XY
    X_sq = np.sum(np.square(X), 1).reshape(-1, 1)
    Y_sq = np.sum(np.square(centroids), 1).reshape(1,-1)
    R = X_sq + Y_sq - 2 * X @ centroids.T
    return np.argmin(R, 1)

def computeCentroids(X, idx, K):
    n = 1
    if X.ndim > 1:
        n = X.shape[1]
    
    C = np.zeros((K,n))
    for k in range(K):
        C[k] = np.mean(X[idx==k], 0)
    return C

def plotDataPoints(X, idx, K):
    # Plot the data
    cmap = plt.get_cmap('jet')
    colors = cmap(idx/max(idx))
    plt.scatter(X[:, 0], X[:, 1], 15, edgecolors='k', marker='o', facecolors=colors, lw=0.5)

def plotProgresskMeans(X, centroids, previous, idx, K, i):
    # Plot the examples
    plotDataPoints(X, idx, K)

    # Plot the centroids as black x's
    plt.plot(centroids[:,0], centroids[:,1], 'x', mec ='k', ms = 5, lw = 5)
    # Plot the history of the centroids with lines
    for j in range(centroids.shape[0]):
        plt.plot([centroids[j,0], previous[j,0]], [centroids[j,1], previous[j,1]], 'b-')
    # Title
    plt.title(f'Iteration number {i+1}')
    plt.ion()
    plt.show()
    plt.pause(0.01)
    
def runkMeans(X, initial_centroids, max_iters, plot_progress=False):
    centroids = initial_centroids
    previous_centroids = centroids
    K = initial_centroids.shape[0]
    for i in range(max_iters):
        print(f'K-Means iteration {i+1}/{max_iters}...')
        idx = findClosestCentroids(X, centroids)
        # Optionally, plot progress here
        if plot_progress is True:
            plotProgresskMeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids
            input('Press enter to continue.\n')
        centroids = computeCentroids(X, idx, K)
    plt.ioff()
    plt.show()
    return centroids, idx

def kMeansInitCentroids(X, K):
    # Randomly reorder the indices of examples
    randidx = np.random.permutation(X.shape[0])
    # Take the first K examples as centroids
    return X[randidx[:K], :]

## ================= Part 1: Find Closest Centroids ====================

print('Finding closest centroids.\n')

# Load an example dataset that we will be using
scriptdir = os.path.dirname(os.path.realpath(__file__))
data = loadmat(scriptdir + '//ex7data2.mat')
X = data['X']

# Select an initial set of centroids
K = 3 # 3 Centroids
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

# Find the closest centroids for the examples using the
# initial_centroids
idx = findClosestCentroids(X, initial_centroids)

print('Closest centroids for the first 3 examples: ')
print(idx[:3])
print('\n(the closest centroids should be 0, 2, 1 respectively)')

input('Program paused. Press enter to continue.\n')

## ===================== Part 2: Compute Means =========================

print('\nComputing centroids means.\n')

#  Compute means based on the closest centroids found in the previous part.
centroids = computeCentroids(X, idx, K)

print('Centroids computed after initial finding of closest centroids: ')
print(centroids)
print('\n(the centroids should be')
print('   [ 2.428301 3.157924 ]')
print('   [ 5.813503 2.633656 ]')
print('   [ 7.119387 3.616684 ]\n')

input('Program paused. Press enter to continue.\n')

## =================== Part 3: K-Means Clustering ======================
print('\nRunning K-Means clustering on example dataset.\n')

# Load an example dataset

# Settings for running K-Means
K = 3
max_iters = 10


initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

# Run K-Means algorithm. The 'true' at the end tells our function to plot
# the progress of K-Means
centroids, idx = runkMeans(X, initial_centroids, max_iters, True)
print('\nK-Means Done.\n')

input('Program paused. Press enter to continue.\n')

## ============= Part 4: K-Means Clustering on Pixels ===============

print('\nRunning K-Means clustering on pixels from an image.\n')

#  Load an image of a bird
A = imread(scriptdir + '//bird_small.png')

# If imread does not work for you, you can try instead
#   load ('bird_small.mat')

A = A / 255.0 # Divide by 255 so that all values are in the range 0 - 1

# Size of the image
img_shape = A.shape

# Reshape the image into an Nx3 matrix where N = number of pixels.
# Each row will contain the Red, Green and Blue pixel values
# This gives us our dataset matrix X that we will use K-Means on.
X = A.reshape(-1, 3)

# Run your K-Means algorithm on this data
# You should try different values of K and max_iters here
K = 16
max_iters = 10

# When using K-Means, it is important the initialize the centroids
# randomly. 
initial_centroids = kMeansInitCentroids(X, K)

# Run K-Means
centroids, idx = runkMeans(X, initial_centroids, max_iters)

input('Program paused. Press enter to continue.\n')

## ================= Part 5: Image Compression ======================

print('\nApplying K-Means to compress an image.\n')

# Find closest cluster members
idx = findClosestCentroids(X, centroids)

X_recovered = centroids[idx,:]

# Reshape the recovered image into proper dimensions
X_recovered = X_recovered.reshape(img_shape)

# Display the original image 
plt.subplot(1, 2, 1)
plt.imshow(A)
plt.title('Original')

# Display compressed image side by side
plt.subplot(1, 2, 2)
plt.imshow(X_recovered)
plt.title(f'Compressed, with {K} colors.')

plt.show()

input('Program paused. Press enter to continue.\n')