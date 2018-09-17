import numpy as np
import matplotlib.pyplot as plt

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

def sigmoid(z):
    return 1/(1+np.exp(-z))
