# Machine Learning: k-Nearest Neighbors (kNN)

This project demonstrates how to implement the **k-Nearest Neighbors (kNN)** algorithm from scratch using Python. It includes:

- Binary classification (Pass/Fail) using height and weight
- Multi-class classification (Species A, B, C) using flower measurements
- Decision boundary visualization using Matplotlib

---

## Files Included

| File Name                        | Description |
|----------------------------------|-------------|
| `knn_binary_classification.py`   | kNN for binary classification with 2 features |
| `knn_multiclass_classification.py` | kNN for multi-class classification with 3 features |
| `knn_visualization.py`           | Visualization of decision boundaries for binary classification |
| `README.md`                      | Project documentation |

---

## Description

This project uses **Euclidean distance** to measure similarity between data points.

### 1. Binary Classification (2 Features)
- Features: `height`, `weight`
- Labels: `Pass=1`, `Fail=0`
- Uses manual kNN algorithm (no `sklearn`)
- Value of `k`: 3

### 2. Multi-Class Classification (3 Features)
- Features: `sepal length`, `sepal width`, `petal length`
- Labels: `Species A=0`, `B=1`, `C=2`
- Value of `k`: 5

### 3. Visualization
- Uses a mesh grid to visualize the **decision boundaries**
- Scatter plot colored by class
- Visualizes how the algorithm separates regions of the feature space

---
