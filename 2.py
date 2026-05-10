from src.data_prepare import load_and_split_wine_data
from src.RandomForest import RandomForest
from src.GridSearch_a2 import grid_search_rf
from sklearn.metrics import f1_score, mean_squared_error
import numpy as np

X_train, X_test, y_train, y_test = load_and_split_wine_data()

param_grid = {
    'n_trees': [10, 30, 50],
    'max_depth': [8, 12, 15],
    'min_samples_split': 5
}

print("Bắt đầu tuning...")
best_params, _ = grid_search_rf(
    RandomForest, X_train, y_train, param_grid)

print("-" * 30)
print(f"Tham số tốt nhất tìm được: {best_params}")

final_rf = RandomForest(**best_params)
final_rf.fit(X_train, y_train)

y_pred = final_rf.predict(X_test)
f1 = f1_score(y_test, y_pred, average='weighted')

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n--- KẾT QUẢ ASSIGNMENT 2 ---")
print(f"F1 Score (Weighted): {f1:.4f}")
print(f"RMSE: {rmse:.4f}")

# --- KẾT QUẢ ASSIGNMENT 2 ---
# F1 Score (Weighted): 0.6553
# RMSE: 0.6889

# Tham số tốt nhất tìm được: {'n_trees': 30, 'max_depth': 15, 'min_samples_split': 5}
