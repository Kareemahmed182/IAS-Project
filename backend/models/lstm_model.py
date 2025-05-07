import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

def reshape_for_lstm(X, timesteps=1):
    return np.reshape(X, (X.shape[0], timesteps, X.shape[1]))

def train_and_evaluate_lstm(X_train, y_train, X_test, y_test, epochs=20, batch_size=32):
    # Reshape inputs for LSTM
    X_train_lstm = reshape_for_lstm(X_train)
    X_test_lstm = reshape_for_lstm(X_test)

    # Build LSTM model (multi-output regression)
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
    model.add(Dense(2))  # 2 outputs: temperature and humidity

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train_lstm, y_train, epochs=epochs, batch_size=batch_size,
              validation_split=0.1, verbose=1, callbacks=[EarlyStopping(patience=3)])

    # Predict
    y_pred = model.predict(X_test_lstm)

    # Save predictions to CSV
    results = pd.DataFrame({
        'Actual_Temperature': y_test.iloc[:, 0],
        'Pred_Temperature': y_pred[:, 0],
        'Actual_Humidity': y_test.iloc[:, 1],
        'Pred_Humidity': y_pred[:, 1],
    })
    results.to_csv("output/predictions_lstm.csv", index=False)

    # Evaluate
    metrics = {
        'MAE_Temperature': mean_absolute_error(y_test.iloc[:, 0], y_pred[:, 0]),
        'RMSE_Temperature': mean_squared_error(y_test.iloc[:, 0], y_pred[:, 0]) ** 0.5,
        'R2_Temperature': r2_score(y_test.iloc[:, 0], y_pred[:, 0]),
        'MAE_Humidity': mean_absolute_error(y_test.iloc[:, 1], y_pred[:, 1]),
        'RMSE_Humidity': mean_squared_error(y_test.iloc[:, 1], y_pred[:, 1]) ** 0.5,
        'R2_Humidity': r2_score(y_test.iloc[:, 1], y_pred[:, 1]),
    }

    return model, metrics
