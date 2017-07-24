from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
import sys

# preparing teaching data
features = np.array([
    # AIPA
    # ABV, IBU
    [6.2, 65],
    [6.2, 80],
    [6.5, 65],
    [6.7, 55],
    [6.8, 75],
    [6.8, 72],
    [6.8, 70],
    [7.0, 90],
    [7.0, 62],
    [7.5, 90],

    # APA
    # ABV, IBU
    [4.8, 50],
    [5.2, 65],
    [5.3, 60],
    [5.3, 60],
    [5.4, 65],
    [5.8, 50],
    [6.0, 30],
    [6.0, 70],
    [6.2, 50],
    [6.6, 45]
])


labels = np.array([
    # AIPA                        APA
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
])


# preparing data for analysis
X = np.array([
    [7.3, 80],  # Expected: AIPA (0)
    [5.5, 35]   # Expected: APA  (1)
])

Y = np.array([
    0, 1
])

# entry point
if __name__ == '__main__':
    try:
        clf = tree.DecisionTreeClassifier()  # acquiring classifier
        clf = clf.fit(features, labels)      # teaching  classifier
        y = clf.predict(X)                   # computing output

        # explain features on chart
        AIPA_X = []
        AIPA_Y = []
        APA_X = []
        APA_Y = []
        for k, f in enumerate(features):
            if k <= 9:
                AIPA_X.append(f[0])
                AIPA_Y.append(f[1])
            else:
                APA_X.append(f[0])
                APA_Y.append(f[1])
        plt.scatter(AIPA_X,  AIPA_Y,  color='red')
        plt.scatter(APA_X,   APA_Y,   color='blue')
        plt.scatter(X[0][0], X[0][1], color='yellow')
        plt.scatter(X[1][0], X[1][1], color='black')
        plt.show()

        # display summary
        print('computed: {0}'.format(y))
        print('expected: {0}'.format(Y))
    except KeyboardInterrupt:
        sys.exit(0)
