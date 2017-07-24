from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np
import sys

# generate sample data
X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()

# add noise to every fifth target value
y[::5] += 3 * (0.5 - np.random.rand(8))

# entry point
if __name__ == '__main__':
    try:
        # Support Vector Machine - Regression
        rbf = svm.SVR(kernel='rbf',    C=1e3, gamma=0.1)
        lin = svm.SVR(kernel='linear', C=1e3)
        pol = svm.SVR(kernel='poly',   C=1e3, degree=2)
        y_rbf = rbf.fit(X, y).predict(X)
        y_lin = lin.fit(X, y).predict(X)
        y_pol = pol.fit(X, y).predict(X)

        # display results
        plt.scatter(X, y,  color='black', s=1,  label='data')
        plt.plot(X, y_rbf, color='red',   lw=1, label='RBF model')
        plt.plot(X, y_lin, color='green', lw=1, label='Linear model')
        plt.plot(X, y_pol, color='blue',  lw=1, label='Polynomial model')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title('Support Vector Machine - Regression')
        plt.legend()
        plt.show()
    except KeyboardInterrupt:
        sys.exit(0)