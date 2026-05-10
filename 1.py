from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, mean_squared_error
from src.DecisionTree import DecisionTree
from src.GridSearch_a1 import grid_search_cv
from src.data_prepare import load_and_split_wine_data

X_train, X_test, y_train, y_test = load_and_split_wine_data()

param_grid = {
    'max_depth': [5, 8, 10, 12, 15, 20],
    'min_samples_split': 5
}

print("Bắt đầu quá trình Tuning...")
best_params, best_f1 = grid_search_cv(
    DecisionTree, X_train, y_train, param_grid)

print("-" * 30)
print(f"Tham số tốt nhất: {best_params}")
print(f"F1-score tối ưu trên tập Train (CV): {best_f1:.4f}")

final_dt = DecisionTree(**best_params)
final_dt.fit(X_train, y_train)

y_pred = final_dt.predict(X_test)

f1 = f1_score(y_test, y_pred, average='weighted')
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"--- Kết quả Assignment 1 ---")
print(f"F1 Score (Weighted): {f1:.4f}")
print(f"RMSE: {rmse:.4f}")
