import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score


def grid_search_cv(model_class, X, y, param_grid, cv=5):
    """
    Hàm dò tìm tham số tối ưu bằng Cross-Validation.
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    best_score = -1
    best_params = {}

    # Duyệt qua các tổ hợp tham số trong lưới
    for depth in param_grid['max_depth']:
        fold_scores = []

        for train_idx, val_idx in kf.split(X):
            # Tách fold theo chỉ mục của KFold
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            # Khởi tạo mô hình với tham số hiện tại
            model = model_class(
                max_depth=depth, min_samples_split=param_grid['min_samples_split'])
            model.fit(X_train_fold, y_train_fold)

            # Dự đoán và tính F1-score weighted cho đa lớp
            y_pred = model.predict(X_val_fold)
            score = f1_score(y_val_fold, y_pred, average='weighted')
            fold_scores.append(score)

        avg_score = np.mean(fold_scores)
        print(f"Thử nghiệm: max_depth={depth} => Avg F1: {avg_score:.4f}")

        if avg_score > best_score:
            best_score = avg_score
            best_params = {'max_depth': depth,
                           'min_samples_split': param_grid['min_samples_split']}

    return best_params, best_score
