def calc_accuracy(y_true, y_predicted):
    correct = sum(yt == yp for yt, yp in zip(y_true, y_predicted))
    return round(100*[correct / len(y_true)][0][0],2)