import numpy as np


class Node:
    """Đại diện cho một nút trong cây"""

    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def _entropy(self, y):
        proportions = np.bincount(y) / len(y)
        return -np.sum([p * np.log2(p) for p in proportions if p > 0])

    def _information_gain(self, y, X_column, threshold):
        parent_entropy = self._entropy(y)

        # Tạo các chỉ mục cho nhánh trái và phải
        left_idxs = np.where(X_column <= threshold)[0]
        right_idxs = np.where(X_column > threshold)[0]

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # Tính Entropy trung bình có trọng số của các con
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        return parent_entropy - child_entropy

    def _best_split(self, X, y):
        best_gain = -1
        split_idx, split_thresh = None, None

        for idx in range(X.shape[1]):
            X_column = X[:, idx]
            thresholds = np.unique(X_column)
            for thr in thresholds:
                gain = self._information_gain(y, X_column, thr)
                if gain > best_gain:
                    best_gain, split_idx, split_thresh = gain, idx, thr
        return split_idx, split_thresh

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Điều kiện dừng: đạt độ sâu tối đa hoặc nhãn thuần nhất
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = np.bincount(y).argmax()
            return Node(value=leaf_value)

        # Tìm điểm chia tốt nhất
        best_feat, best_thr = self._best_split(X, y)

        # Xây dựng các nhánh con
        left_idxs = np.where(X[:, best_feat] <= best_thr)[0]
        right_idxs = np.where(X[:, best_feat] > best_thr)[0]

        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thr, left, right)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
