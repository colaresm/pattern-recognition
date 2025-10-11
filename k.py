
# --- Show results ---
plt.figure()
plt.plot(p_values, [results[p]['mean'] for p in p_values], marker='o', color='black')
plt.xlabel("p value")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.show()

# --- Confusion matrix for bes p value ---
best_p = max(results, key=lambda p: results[p]['mean'])
best_round = results[best_p]['best_round_index']

y_true_best = results[best_p]['true_labels'][best_round]
y_pred_best = results[best_p]['predictions'][best_round]

plot_confusion_matrix(y_true_best, y_pred_best, f"Best Confusion Matrix (p={best_p})")

print(f"\nBest value of p: {best_p}, the mean accuracy is {results[best_p]['mean']:.4f}")
