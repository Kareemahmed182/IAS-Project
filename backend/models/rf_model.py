from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_and_evaluate_rf(X_train, y_train, X_test, y_test):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    # Combine predictions with original features
    results = X_test.copy()
    results['Actual_Temperature'] = y_test.iloc[:, 0]
    results['Pred_Temperature'] = y_pred[:, 0]
    results['Actual_Humidity'] = y_test.iloc[:, 1]
    results['Pred_Humidity'] = y_pred[:, 1]

    results.to_csv("output/predictions_rf.csv", index=False)

    # Evaluation metrics
    metrics = {
        'MAE_Temperature': mean_absolute_error(y_test.iloc[:, 0], y_pred[:, 0]),
        'RMSE_Temperature': mean_squared_error(y_test.iloc[:, 0], y_pred[:, 0]) ** 0.5,
        'R2_Temperature': r2_score(y_test.iloc[:, 0], y_pred[:, 0]),
        'MAE_Humidity': mean_absolute_error(y_test.iloc[:, 1], y_pred[:, 1]),
        'RMSE_Humidity': mean_squared_error(y_test.iloc[:, 1], y_pred[:, 1]) ** 0.5,
        'R2_Humidity': r2_score(y_test.iloc[:, 1], y_pred[:, 1]),
    }

    return rf, metrics
