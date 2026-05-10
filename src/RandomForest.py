import numpy as np
from DecisionTreeRF import DecisionTree


class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None, random_state=42):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.random_state = random_state  # Thêm seed để cố định kết quả
        self.trees = []

    def _bootstrap_samples(self, X, y, seed):
        np.random.seed(seed)  # Cố định mẫu cho từng cây
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def fit(self, X, y):
        self.trees = []
        for i in range(self.n_trees):
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.n_features
            )
            # Mỗi cây có một seed riêng dựa trên random_state gốc
            X_sample, y_sample = self._bootstrap_samples(
                X, y, self.random_state + i)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        # Biểu quyết đa số (Majority Voting)
        y_pred = [np.bincount(sample_preds).argmax()
                  for sample_preds in tree_preds]
        return np.array(y_pred)
