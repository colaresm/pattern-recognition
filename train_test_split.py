import random

def train_test_split(X, y, test_size=0.2, seed=None):
    y = process_labels(y)
    if seed is not None:
        random.seed(seed)
    assert len(X) == len(y), "X e y devem ter o mesmo tamanho."
    n = len(X)
    n_test = int(n * test_size)
    indices = list(range(n))
    random.shuffle(indices)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    X_train = [X[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]
    y_train = [y[i] for i in train_indices]
    y_test = [y[i] for i in test_indices]
    return X_train, X_test, y_train, y_test

def process_labels(labels):
    processed_labels = []
    for label in labels:
        processed_labels.append(label[0])
    return processed_labels