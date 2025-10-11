from train_test_split import train_test_split
from dataset import load_data
from metrics import calc_accuracy
from nn_clf import NearestNeighbor
from metrics import stats, plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import time

start = time.time()
X, y = load_data()
end = time.time()
print("Execution train time:", end - start, "seconds")

X_train, X_test, y_train, y_test = train_test_split(X, y)

#### NN CLASSIFIER #####
p_values = [0.5, 2/3, 1, 3/2, 2, 5/2]
test_sizes = [0.2, 0.3, 0.5, 0.7, 0.8]

results = {}

for i, test_size in enumerate(test_sizes):
    for j, p in enumerate(p_values):
        accs = []
        predictions_per_round = []
        true_labels_per_round = []
        for _ in range(10):   
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
            model = NearestNeighbor(p=p)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            predictions_per_round.append(predictions)
            true_labels_per_round.append(y_test)
            acc = calc_accuracy(y_test, predictions)
            accs.append(acc)

        mean_acc, std_acc, max_acc, min_acc, median_acc = stats(accs)
        best_round_index = np.argmax(accs)
        worse_round_index = np.argmin(accs)

 
        results[(i, j)] = {
            "mean": mean_acc,
            "std": std_acc,
            "max": max_acc,
            "min": min_acc,
            "median": median_acc,
            "best_round_index": best_round_index,
            "worse_round_index": worse_round_index,
            "accs": accs,
            "true_labels": true_labels_per_round,
            "predictions": predictions_per_round,
            "test_size": test_size,
            "p": p
        }

        print(f"test_size={test_size}, p={p}: Mean={mean_acc:.4f}, Std={std_acc:.4f}, Max={max_acc:.4f}, Min={min_acc:.4f}, Median={median_acc:.4f}")

# --- Show results ---
unique_test_sizes = sorted(set([results[key]["test_size"] for key in results.keys()]))

plt.figure(figsize=(10, 6))
for test_size in unique_test_sizes:
    x_vals = []
    y_vals = []
    for (i, j), data in results.items():
        if data["test_size"] == test_size:
            x_vals.append(data["p"])
            y_vals.append(data["mean"])
    x_vals, y_vals = zip(*sorted(zip(x_vals, y_vals)))
    plt.plot(x_vals, y_vals, marker='o', label=f"test_size={test_size}")

plt.xlabel("p")
plt.ylabel("Acurácia média")
plt.grid(True)
plt.legend(title="Test Size")
plt.tight_layout()
plt.show()

# --- Confusion matrix for bes p value ---
best_p = max(results, key=lambda p: results[p]['mean'])
best_round = results[best_p]['best_round_index']

y_true_best = results[best_p]['true_labels'][best_round]
y_pred_best = results[best_p]['predictions'][best_round]

plot_confusion_matrix(y_true_best, y_pred_best, f"Best Confusion Matrix (p={best_p})")

print(f"\nBest value of p: {best_p}, the mean accuracy is {results[best_p]['mean']:.4f}")
