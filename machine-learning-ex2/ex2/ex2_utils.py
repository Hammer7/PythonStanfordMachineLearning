import matplotlib.pyplot as plt
import numpy as np

def mapFeature(X1, X2):
    m = X1.size
    out = np.ones((m, 1))
    degree = 6
    for i in range(1, degree+1):
        for j in range(i+1):
            out = np.c_[out, np.power(X1, i-j) * np.power(X2, j)]
    return out

def plotData(x, y):
    pos = np.where(y==1)
    neg = np.where(y==0)
    plt.plot(x[pos[0],0], x[pos[0],1], 'b+', lw = 1, ms = 5)
    plt.plot(x[neg[0],0], x[neg[0],1], 'ro', lw = 1, ms = 5)

def plotDecisionBoundary(theta, X, y):
    if X.shape[1] <= 3:
        plot_x = np.array([np.amin(X[:,1])-2,  np.amax(X[:,1])+2])
        plot_y = -1/theta[2] * (theta[0] + plot_x * theta[1])
        plotData(X[:, 1:],y)
        plt.plot(plot_x, plot_y, 'g-')
    else:
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)

        z = np.zeros((u.size, v.size))
        # Evaluate z = theta*x over the grid
        for i in range(u.size):
            for j in range(v.size):
                z[j, i] = mapFeature(u[i], v[j]) @ theta
        #Plot z = 0
        #Notice you need to specify the range [0, 0]
        plotData(X[:, 1:],y)
        plt.contour(u, v, z, [0])

def sigmoid(z):
    return 1/(1+np.exp(-z))

def predict(theta, X):
    p = sigmoid(X @ theta)
    return p >= 0.5