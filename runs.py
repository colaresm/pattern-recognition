import numpy as np
from train_test_split import train_test_split
from metrics import calc_accuracy, stats
from nn_clf import NearestNeighbor
from mdc_clf import MinimumDistanceCentroid
from mc_clf import MaximumCorrelation
import time

def NN(X, y, test_sizes, p_values, rounds=10):
    results = {}
    for i, test_size in enumerate(test_sizes):
        for j, p in enumerate(p_values):
            accs = []
            test_times = []
            predictions_per_round = []
            true_labels_per_round = []
            for _ in range(rounds):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
                model = NearestNeighbor(p=p)
                model.fit(X_train, y_train)
                start  = time.time()
                predictions = model.predict(X_test)
                end = time.time()
                test_time = round(end - start,4)
                test_times.append(round(test_time,5))
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
                "p": p,
                "test_time":np.mean(test_times)
            }

    return results

def MDC(X, y, test_sizes, rounds=10,is_robust_version=False):
    results = {}
    p_values = [0]
    for i, test_size in enumerate(test_sizes):
        for j, p in enumerate(p_values):
            accs = []
            test_times = []
            predictions_per_round = []
            true_labels_per_round = []
            for _ in range(rounds):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
                model = MinimumDistanceCentroid(1)
                model.fit(X_train, y_train,is_robust_version)
                start  = time.time()
                predictions = model.predict(X_test)
                end = time.time()
                test_time = round(end - start,4)
                test_times.append(round(test_time,5))
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
                "p": p,
                "test_time":np.mean(test_times)
            }

    return results

def mc(X, y, test_sizes, rounds=10):
    results = {}
    p_values = [2]
    for i, test_size in enumerate(test_sizes):
        for j, p in enumerate(p_values):
            accs = []
            test_times = []
            predictions_per_round = []
            true_labels_per_round = []
            for _ in range(rounds):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
                model = MaximumCorrelation(1)
                model.fit(X_train, y_train)
                start  = time.time()
                predictions = model.predict(X_test)
                end = time.time()
                test_time = round(end - start,4)
                test_times.append(round(test_time,5))
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
                "p": p,
                "test_time":np.mean(test_times)
            }

    return results