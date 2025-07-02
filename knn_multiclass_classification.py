import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Dataset
data = np.array([
    [5.1, 3.5, 1.4, 0],
    [4.9, 3.0, 1.3, 0],
    [5.0, 3.4, 1.5, 0],
    [7.0, 3.2, 4.7, 1],
    [6.4, 3.2, 4.5, 1],
    [6.9, 3.1, 4.9, 1],
    [5.5, 2.3, 4.0, 2],
    [6.5, 2.8, 4.6, 2],
    [5.7, 2.8, 4.1, 2],
    [6.3, 3.3, 6.0, 2],
    [5.8, 2.7, 5.1, 2],
    [6.1, 3.0, 4.8, 2]
])

X = data[:, :3]
y = data[:, 3]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def knn_predict(X_train, y_train, x_test, k):
    distances = [euclidean_distance(x_test, x_train) for x_train in X_train]
    k_indices = np.argsort(distances)[:k]
    k_nearest_labels = [y_train[i] for i in k_indices]
    most_common = Counter(k_nearest_labels).most_common(1)
    return most_common[0][0]

# Predict
k = 5
predictions = [knn_predict(X_train, y_train, x, k) for x in X_test]
print("Accuracy:", accuracy_score(y_test, predictions))
