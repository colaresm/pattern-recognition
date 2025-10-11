import numpy as np

class MinimumDistanceCentroid:
    def __init__(self, p=2):
        self.p = p
        self.X_train = None
        self.y_train = None
        self.classes_ = None
        self.centroids = {}
        self.features_per_class = {}
        self.is_robust_version = False
    
    def minkowski_distance(self, x1, x2):
        if len(x1) != len(x2):
            raise ValueError("The vectors must have the same length.")
        if self.is_robust_version:
            self.p = 1
        else:
            self.p = 2
            
        s = sum(abs(a - b) ** self.p for a, b in zip(x1, x2))
        return s ** (1 / self.p)
    
    def calc_centroid(self, X):
        if self.is_robust_version:
            return np.median(X,axis=0)
        return np.mean(X, axis=0)

    def fit(self, X_train, y_train,is_robust_version):
        self.X_train = X_train
        self.y_train = y_train
        self.is_robust_version = is_robust_version
        self.classes_ = np.unique(self.y_train)
    
        for class_ in self.classes_:
            ids = np.where(np.array(y_train) == class_)[0]   
            self.features_per_class[class_] = np.array(X_train)[ids]
            self.centroids[class_] = self.calc_centroid(self.features_per_class[class_])
    
    def _mdc(self, X_new):
        dists = []
        for cls in self.classes_:
            centroid = self.centroids[cls]
            dist = self.minkowski_distance(centroid, X_new)
            dists.append(dist)
        index = np.argmin(dists)
        return self.classes_[index]
      
    def predict(self, X_test):
        predictions = []
        for X in X_test:
            predictions.append(self._mdc(X))
        return predictions
