import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score


def grid_search_rf(model_class, X, y, param_grid, cv=5):
    """
    Dò tìm n_trees và max_depth tối ưu cho Random Forest.
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    best_score = -1
    best_params = {}

    for n_trees in param_grid['n_trees']:
        for depth in param_grid['max_depth']:
            fold_scores = []

            for train_idx, val_idx in kf.split(X):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]

                # Khởi tạo mô hình RF
                model = model_class(
                    n_trees=n_trees,
                    max_depth=depth,
                    min_samples_split=param_grid['min_samples_split']
                )
                model.fit(X_train_fold, y_train_fold)

                y_pred = model.predict(X_val_fold)
                score = f1_score(y_val_fold, y_pred, average='weighted')
                fold_scores.append(score)

            avg_score = np.mean(fold_scores)
            print(
                f"Thử nghiệm: n_trees={n_trees}, depth={depth} => F1: {avg_score:.4f}")

            if avg_score > best_score:
                best_score = avg_score
                best_params = {'n_trees': n_trees, 'max_depth': depth,
                               'min_samples_split': param_grid['min_samples_split']}

    return best_params, best_score
