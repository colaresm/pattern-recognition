import numpy as np

class MinimumDistanceCentroid:
    def __init__(self, p=0.5):
        self.p = p
        self.X_train = None
        self.y_train = None
        
    def minkowski_distance(self, x1, x2):
        if len(x1) != len(x2):
            raise ValueError("The vectors must have the same length.")
        s = sum(abs(a - b) ** self.p for a, b in zip(x1, x2))
        return s ** (1 / self.p)
    
    def calc_centroid(self, x):
        N = np.len(x)
        centroid = sum(x)/N
        return centroid

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def _mdc(self, X_new):
        dists = [self.minkowski_distance(X, X_new) for X in self.X_train]
        index = np.argmin(dists)
        return self.y_train[index]

    def predict(self, X_test):
        predictions = []
        for X in X_test:
            predictions.append(self._nn(X))
      
        return predictions
