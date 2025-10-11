import numpy as np

class MaximumCorrelation:
    def __init__(self, p=2):
        self.p = p
        self.X_train = None
        self.y_train = None
        self.classes_ = None
        self.centroids = {}
        self.features_per_class = {}
    
    def calc_centroid(self, X):
        return np.mean(X, axis=0)

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.classes_ = np.unique(self.y_train)
    
        for class_ in self.classes_:
            ids = np.where(np.array(y_train) == class_)[0]   
            self.features_per_class[class_] = np.array(X_train)[ids]
            self.centroids[class_] = self.calc_centroid(self.features_per_class[class_])
    
    def _mc(self, X_new):
        prods = []
        X_new = np.array(X_new) / np.linalg.norm(X_new)  

        for cls in self.classes_:
            centroid = np.array(self.centroids[cls])
            centroid = centroid / np.linalg.norm(centroid)
            prod = np.dot(centroid, X_new)  
            prods.append(prod)

        index = np.argmax(prods)  
        return self.classes_[index]

      
    def predict(self, X_test):
        predictions = []
        for X in X_test:
            predictions.append(self._mc(X))
        return predictions
