import math
import random
import numpy as np

class SVM:
    
    def svmTrain(self, X, Y, C, kernelFunction, tol = 1e-3, max_passes = 5):
        m = X.shape[0]
        K = np.zeros((m,m))
        if kernelFunction.__name__ is 'linearKernel':
            #Vectorized computation for the Linear Kernel
            #This is equivalent to computing the kernel on every pair of examples
            K = X @ X.T
        elif kernelFunction.__name__ is 'gaussianKernel':
            X2 = np.sum(np.square(X), 1).reshape(-1,1)
            K = X2 + X2.T - 2 * (X @ X.T)
            K = np.power(kernelFunction(np.ones(1), np.zeros(1)), K)
        else:
            for i in range(m):
                for j in range(i,m):
                    K[i,j] = kernelFunction(X[i,:], X[j,:])
                    K[j,i] = K[i,j] #the matrix is symmetric
        print('\nTraining...')
        dots = 12
        passes = 0
        # Map 0 to -1
        Y[Y==0] = -1
        alphas = np.zeros(m)
        b = 0.0
        while passes < max_passes:
            num_changed_alphas = 0
            for i in range(m):
                # Calculate Ei = f(x[i]) - y[i] using (2). 
                # E[i] = b + sum (X[i, :] * (repmat(alphas*Y,1,n)*X).T) - Y[i]
                E_i = b + (alphas * Y @ K[:,i]) - Y[i]
                if ((Y[i] * E_i < -tol) and (alphas[i] < C)) or ((Y[i] * E_i > tol) and (alphas[i] > 0)):
                    # In practice, there are many heuristics one can use to select
                    # the i and j. In this simplified code, we select them randomly.
                    j = random.randint(0, m-1)
                    while j == i:  # Make sure i \neq j
                        j = random.randint(0, m-1)
                    # Calculate Ej = f(x[j]) - y[j] using (2).
                    E_j = b + (alphas * Y @ K[:,j]) - Y[j]
                    # Save old alphas
                    alpha_i_old = alphas[i]
                    alpha_j_old = alphas[j]
                    
                    # Compute L and H by (10) or (11). 
                    if Y[i] == Y[j]:
                        L = max(0, alphas[j] + alphas[i] - C)
                        H = min(C, alphas[j] + alphas[i])
                    else:
                        L = max(0, alphas[j] - alphas[i])
                        H = min(C, C + alphas[j] - alphas[i])
                    
                    if L == H:
                        # continue to next i. 
                        continue

                    # Compute eta by (14).
                    eta = 2 * K[i,j] - K[i,i] - K[j,j]
                    if eta >= 0:
                        # continue to next i. 
                        continue

                    # Compute and clip new value for alpha j using (12) and (15).
                    alphas[j] = alphas[j] - (Y[j] * (E_i - E_j) / eta)
                    
                    # Clip
                    alphas[j] = min(H, alphas[j])
                    alphas[j] = max(L, alphas[j])
                    
                    # Check if change in alpha is significant
                    if abs(alphas[j] - alpha_j_old) < tol:
                        # continue to next i. 
                        # replace anyway
                        alphas[j] = alpha_j_old
                        continue

                    # Determine value for alpha i using (16). 
                    alphas[i] = alphas[i] + Y[i] * Y[j] * (alpha_j_old - alphas[j])
                    
                    # Compute b1 and b2 using (17) and (18) respectively. 
                    b1 = b - E_i - Y[i] * (alphas[i] - alpha_i_old) *  K[i,i] - Y[j] * (alphas[j] - alpha_j_old) *  K[i,j]
                    b2 = b - E_j - Y[i] * (alphas[i] - alpha_i_old) *  K[i,j] - Y[j] * (alphas[j] - alpha_j_old) *  K[j,j]

                    # Compute b by (19). 
                    if alphas[i] > 0 and alphas[i] < C:
                        b = b1
                    elif alphas[j] > 0 and alphas[j] < C:
                        b = b2
                    else:
                        b = (b1+b2)/2.0

                    num_changed_alphas = num_changed_alphas + 1

            if num_changed_alphas == 0:
                passes = passes + 1
            else:
                passes = 0
            
            print('.', end='', flush=True)
            dots = dots + 1
            if dots > 78:
                dots = 0
                print('')

        print(' Done! \n')
        # Save the model
        idx = (alphas > 0)
        self.X = X[idx, :]
        self.y = Y[idx]
        self.kernelFunction = kernelFunction
        self.b = b
        self.alphas= alphas[idx]
        #self.w = (alphas * Y @ X).T
        self.w = X.T @ (Y * alphas)

    def svmPredict(self, X):
        m = X.shape[0]
        p = np.zeros(m)
        pred = np.zeros(m)
        if self.kernelFunction.__name__ is 'linearKernel':
            # We can use the weights and bias directly if working with the 
            # linear kernel
            p = X @ self.w + self.b
        elif self.kernelFunction.__name__ is 'gaussianKernel':
            # Vectorized RBF Kernel
            # This is equivalent to computing the kernel on every pair of examples
            X1 = np.sum(np.square(X), 1).reshape(-1,1)
            X2 = np.sum(np.square(self.X), 1).reshape(-1,1).T
            K = X1 + X2 - 2 * X @ self.X.T
            K = np.power(self.kernelFunction(np.ones(1), np.zeros(1)), K) * self.y.T * self.alphas.T
            p = np.sum(K, 1)
        else:
            for i in range(m):
                prediction = 0
                for j in range(self.X.shape[0]):
                    prediction += self.alphas[j] * self.y[j] * self.kernelFunction(X[i,:],self.X[j,:])
                p[i] = prediction + self.b
        #convert predictions into 0 / 1
        pred[p >= 0] = 1
        pred[p <  0] = 0
        return pred
