from sklearn import datasets
from sklearn import svm
import matplotlib.pyplot as plt
import sys

# acquiring example digits images
data_set = datasets.load_digits()

# preparing data to teach
f = data_set.images[:1000]  # features
l = data_set.target[:1000]  # labels

# preparing data for analysis
X = data_set.images[1001:1036]  # features
Y = data_set.target[1001:1036]  # expected labels

# reshaping data from two dimensions into one dimension
f_1D = f.reshape(len(f), -1)
X_1D = X.reshape(len(X), -1)

# entry point
if __name__ == '__main__':
    try:
        # Support Vector Machine Classifier
        clf = svm.SVC(gamma=0.001)  # acquiring classifier
        clf = clf.fit(f_1D, l)      # teaching  classifier
        y = clf.predict(X_1D)       # computing output

        # display summary
        print('computed digits: {0}'.format(y))
        print('expected digits: {0}'.format(Y))

        # display images
        for k, v in enumerate(y):
            original_digit = X[k]
            computed_digit = str(v)

            plt.subplot(5, 7, k+1)
            plt.axis('off')
            plt.imshow(
                original_digit,
                cmap=plt.cm.gray_r,
                interpolation='nearest'
            )
            plt.title(computed_digit)
        plt.show()
    except KeyboardInterrupt:
        sys.exit(0)
