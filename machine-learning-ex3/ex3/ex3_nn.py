import os
import numpy as np
from scipy.io import loadmat
from ex3_utils import displayData, sigmoid

## Machine Learning Online Class - Exercise 3 | Part 2: Neural Networks

def predict(Theta1, Theta2, X):
    m = X.shape[0]
    X = np.c_[np.ones(m), X]
    A2 = np.c_[np.ones(m), sigmoid(X @ Theta1.T)]
    pred = np.argmax(sigmoid(A2 @ Theta2.T), axis=1) #A3
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
weight = loadmat(scriptdir + '//ex3weights.mat')
Theta1 = weight['Theta1']
Theta2 = weight['Theta2']

# ================= Part 3: Implement Predict =================
#  After training the neural network, we would like to use it to predict
#  the labels. You will now implement the "predict" function to use the
#  neural network to predict the labels of the training set. This lets
#  you compute the training set accuracy.

pred = predict(Theta1, Theta2, X)

print(f'\nTraining Set Accuracy: {np.mean(pred == y) * 100}\n', )

input('Program paused. Press enter to continue.\n')

#  To give you an idea of the network's output, you can also run
#  through the examples one at the a time to see what it is predicting.

#  Randomly permute examples

for i in np.random.permutation(m):
    # Display    
    print('\nDisplaying Example Image')
    displayData(X[np.newaxis, i])

    pred = predict(Theta1, Theta2, X[np.newaxis, i])
    print(f'\nNeural Network Prediction: {pred[0]} (digit {np.mod(pred[0], 10)})\n')
    print(f'Actual y label : {np.mod(y[i], 10)}\n')
    # Pause with quit option
    s = input('Paused - press enter to continue, q to exit:')
    if s == 'q':
        break