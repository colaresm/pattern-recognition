from train_test_split import train_test_split
from dataset import load_data
from metrics import calc_accuracy
import time

start = time.time()
X,y = load_data()
end = time.time()
print("Execution train time:", end - start, "seconds")

X_train, X_test, y_train, y_test = train_test_split(X,y)

def minkowski_distance(x1, x2, p=0.5):
    if len(x1) != len(x2):
        raise ValueError("The vectors must have the same length.")
    s = sum(abs(a - b) ** p for a, b in zip(x1, x2))
    return s ** (1 / p)

def nn(X_new):
    dists = []
    for X in X_train:
        dist = minkowski_distance(X,X_new)
        dists.append(dist)
    index = dists.index(min(dists))
    return y_train[index]

def predict_data():
    predictions = []
    start = time.time()
    for X_to_test in X_test:
        prediction = nn(X_to_test)
        predictions.append(prediction)
    end = time.time()
    print("Execution test time:", end - start, "seconds")
    return predictions

y_predicted = predict_data()
print(calc_accuracy(y_test, y_predicted))
