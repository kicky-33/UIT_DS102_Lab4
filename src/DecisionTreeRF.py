import numpy as np


class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2, n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features  # Số lượng đặc trưng ngẫu nhiên được chọn tại mỗi nút
        self.root = None

    def _entropy(self, y):
        proportions = np.bincount(y) / len(y)
        return -np.sum([p * np.log2(p) for p in proportions if p > 0])

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None

        for idx in feat_idxs:  # Chỉ lặp qua các đặc trưng được chọn ngẫu nhiên
            X_column = X[:, idx]
            thresholds = np.unique(X_column)
            for thr in thresholds:
                gain = self._information_gain(y, X_column, thr)
                if gain > best_gain:
                    best_gain, split_idx, split_thresh = gain, idx, thr
        return split_idx, split_thresh

    def _information_gain(self, y, X_column, threshold):
        parent_entropy = self._entropy(y)
        left_idxs = np.where(X_column <= threshold)[0]
        right_idxs = np.where(X_column > threshold)[0]
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n = len(y)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (len(left_idxs) / n) * e_l + \
            (len(right_idxs) / n) * e_r
        return parent_entropy - child_entropy

    def fit(self, X, y):
        # Nếu n_features không được chỉ định, mặc định lấy sqrt(n_features)
        n_total_features = X.shape[1]
        self.n_features = int(np.sqrt(n_total_features)) if not self.n_features else min(
            n_total_features, self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_total_feats = X.shape

        # 1. Chốt chặn an toàn: Nếu không có mẫu nào lọt vào nhánh này
        if n_samples == 0:
            return None

        # Tính số lượng nhãn duy nhất
        n_labels = len(np.unique(y))

        # 2. Điều kiện dừng (Stopping Criteria)
        # - Đạt độ sâu tối đa (max_depth)
        # - Nhãn hoàn toàn thuần nhất (n_labels == 1)
        # - Số lượng mẫu còn lại quá ít (min_samples_split)
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            # Trả về nút lá với nhãn phổ biến nhất
            leaf_value = np.bincount(y).argmax()
            return {"value": leaf_value}

        # 3. Random Feature Selection
        # Chọn ngẫu nhiên một tập con các đặc trưng để tìm điểm chia
        feat_idxs = np.random.choice(
            n_total_feats, self.n_features, replace=False)

        # Tìm điểm chia tốt nhất từ tập đặc trưng ngẫu nhiên
        best_feat, best_thr = self._best_split(X, y, feat_idxs)

        # Nếu không tìm thấy điểm chia nào làm tăng Information Gain
        if best_feat is None:
            return {"value": np.bincount(y).argmax()}

        # 4. Thực hiện chia nhánh
        left_idxs = np.where(X[:, best_feat] <= best_thr)[0]
        right_idxs = np.where(X[:, best_feat] > best_thr)[0]

        # 5. Kiểm tra tính hợp lệ của nhánh chia
        # Nếu việc chia nhánh tạo ra một bên rỗng, hãy biến nút hiện tại thành nút lá
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return {"value": np.bincount(y).argmax()}

        # 6. Đệ quy xây dựng các cây con (Recursive partitioning)
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)

        return {
            "feature": best_feat,
            "threshold": best_thr,
            "left": left,
            "right": right
        }

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x, node):
        if "value" in node:
            return node["value"]
        if x[node["feature"]] <= node["threshold"]:
            return self._traverse(x, node["left"])
        return self._traverse(x, node["right"])
