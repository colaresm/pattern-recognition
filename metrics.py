import numpy as np

def calc_accuracy(y_true, y_predicted):
    correct = sum(yt == yp for yt, yp in zip(y_true, y_predicted))
    return round(100*[correct / len(y_true)][0][0],2)



def stats(accs):
    mean_acc = np.mean(accs), 
    std_acc = np.std(accs), 
    max_acc = np.max(accs), 
    min_acc = np.min(accs), 
    median_acc = np.median(accs)
    return float(mean_acc[0]), float(std_acc[0]),  float(max_acc[0]), float(min_acc[0]), float(median_acc)