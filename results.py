import matplotlib.pyplot as plt
from metrics import plot_confusion_matrix

def show_results(results):
    unique_test_sizes = sorted(set([results[key]["test_size"] for key in results.keys()]))
    plt.figure(figsize=(10, 6))
    for test_size in unique_test_sizes:
        x_vals = []
        y_vals = []
        for  (i,j),data in results.items():
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
    max_mean = -float('inf')
    best_p = None
    i,j = None,None
    for (a, b), data in results.items():
        if data["mean"] > max_mean:
            max_mean = data["mean"]
            best_p = data["p"]
            i,j = a,b

    best_results = results[(i,j)]
    plot_confusion_matrix(best_results["true_labels"][0], best_results["predictions"][0], f"Best Confusion Matrix (p={best_p}) with test_size = {best_results["test_size"]}")
    # --- Confusion matrix for worst p value ---
    min_mean = float('inf')
    worst_p = None
    i_worst, j_worst = None, None

    for (a, b), data in results.items():
        if data["mean"] < min_mean:
            min_mean = data["mean"]
            worst_p = data["p"]
            i_worst, j_worst = a, b

    worst_results = results[(i_worst, j_worst)]

    plot_confusion_matrix(
        worst_results["true_labels"][0], 
        worst_results["predictions"][0], 
        f'Worst Confusion Matrix (p={worst_p}) with test_size = {worst_results["test_size"]}'
    )

    print(f"\nBest value of p: {best_p}, the mean accuracy is {best_results['mean']:.4f}; test time = {best_results["test_time"]}s")
