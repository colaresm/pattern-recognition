from train_test_split import train_test_split
from dataset import load_data
from metrics import calc_accuracy
from nn_clf import NearestNeighbor
from metrics import stats, plot_confusion_matrix
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
predictions_per_round = []
true_labels_per_round = []
for i in range(0,100):
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    model = NearestNeighbor(p=2)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    predictions_per_round.append(predictions)
    true_labels_per_round.append(y_test)
    acc = calc_accuracy(y_test,predictions)
    accs.append(acc)

mean_acc, std_acc, max_acc, min_acc, median_acc = stats(accs)
best_round_index = np.argmax(accs)
worse_round_index = np.argmin(accs)


print(f"Mean: {mean_acc:.4f}, Std: {std_acc:.4f}, Max: {max_acc:.4f}, Min: {min_acc:.4f}, Median: {median_acc:.4f}")
plt.figure()
plt.plot(accs,color="black")
plt.xlabel("Iteration")
plt.ylabel("Accuracy (%)")
plt.show()

### confusion matrix ####
y_true = true_labels_per_round[best_round_index]
y_predicted = predictions_per_round[best_round_index]
plot_confusion_matrix(y_true,y_predicted,"Best Confusion Matrix")

y_true = true_labels_per_round[worse_round_index]
y_predicted = predictions_per_round[worse_round_index]
plot_confusion_matrix(y_true,y_predicted,"Worse Confusion Matrix")