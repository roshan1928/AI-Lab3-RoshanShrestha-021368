import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Dataset
data = np.array([
    [165, 60, 1],
    [170, 65, 1],
    [160, 55, 0],
    [175, 70, 1],
    [155, 50, 0],
    [168, 62, 1],
    [162, 58, 0],
    [172, 68, 1],
    [158, 53, 0],
    [167, 61, 1]
])

X = data[:, :2]
y = data[:, 2]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def knn_predict(X_train, y_train, x_test, k):
    distances = [euclidean_distance(x_test, x_train) for x_train in X_train]
    k_indices = np.argsort(distances)[:k]
    k_nearest_labels = [y_train[i] for i in k_indices]
    most_common = Counter(k_nearest_labels).most_common(1)
    return most_common[0][0]

# Test
k = 3
predictions = [knn_predict(X_train, y_train, x, k) for x in X_test]
print("Accuracy:", accuracy_score(y_test, predictions))
