import numpy as np
from src.data_prepare import load_and_split_wine_data  # Sử dụng module bạn đã tách
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, mean_squared_error

X_train, X_test, y_train, y_test = load_and_split_wine_data()

# Decision Tree
print("Bắt đầu tuning cho Decision Tree...")
dt_params = {
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}

dt_grid = GridSearchCV(DecisionTreeClassifier(
    random_state=42), dt_params, cv=5, scoring='f1_weighted')
dt_grid.fit(X_train, y_train)
best_dt = dt_grid.best_estimator_

# Random Forest
print("Bắt đầu tuning cho Random Forest...")
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 15, 20],
    'max_features': ['sqrt', 'log2']
}

rf_grid = GridSearchCV(RandomForestClassifier(
    random_state=42), rf_params, cv=5, scoring='f1_weighted')
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_


def evaluate(model, name):
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"\n[{name}]")
    print(
        f"Tham số tốt nhất: {dt_grid.best_params_ if name == 'Decision Tree' else rf_grid.best_params_}")
    print(f"F1 Score: {f1:.4f}")
    print(f"RMSE: {rmse:.4f}")


evaluate(best_dt, "Decision Tree")
evaluate(best_rf, "Random Forest")
