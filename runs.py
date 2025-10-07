from train_test_split import train_test_split
from dataset import load_data
from metrics import calc_accuracy
from nn_clf import NearestNeighbor
from metrics import stats
import matplotlib.pyplot as plt
import numpy as np
import time

start = time.time()
X,y = load_data()
end = time.time()
print("Execution train time:", end - start, "seconds")

X_train, X_test, y_train, y_test = train_test_split(X,y)

#### NN CLASSIFIFER #####
accs = []
for i in range(0,100):
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    model = NearestNeighbor(p=0.5)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    acc = calc_accuracy(y_test,predictions)
    accs.append(acc)

mean_acc, std_acc, max_acc, min_acc, median_acc = stats(accs)

print(f"Mean: {mean_acc:.4f}, Std: {std_acc:.4f}, Max: {max_acc:.4f}, Min: {min_acc:.4f}, Median: {median_acc:.4f}")
plt.figure()
plt.plot(accs)
plt.show()