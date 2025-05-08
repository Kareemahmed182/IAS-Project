import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def create_sequences(X, y, timesteps=20):
    Xs, ys = [], []
    for i in range(len(X) - timesteps):
        Xs.append(X[i:i+timesteps])
        ys.append(y[i+timesteps])
    return np.array(Xs), np.array(ys)

def train_and_evaluate_lstm(X_train, y_train, X_test, y_test, epochs=50, batch_size=32, timesteps=20):
    # Normalize input and output
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)

    # Create sequences for LSTM
    X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, timesteps)
    X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, timesteps)

    # Build stacked LSTM model
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(2))  # Output: temperature & humidity

    model.compile(optimizer='adam', loss='mae')
    model.fit(X_train_seq, y_train_seq, epochs=epochs, batch_size=batch_size,
              validation_split=0.1, verbose=1, callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])

    # Predict and inverse transform
    y_pred_scaled = model.predict(X_test_seq)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_test_actual = y_scaler.inverse_transform(y_test_seq)

    # Save predictions to CSV
    results = pd.DataFrame({
        'Actual_Temperature': y_test_actual[:, 0],
        'Pred_Temperature': y_pred[:, 0],
        'Actual_Humidity': y_test_actual[:, 1],
        'Pred_Humidity': y_pred[:, 1],
    })
    results.to_csv("output/predictions_lstm.csv", index=False)

    # Evaluate
    metrics = {
        'MAE_Temperature': mean_absolute_error(y_test_actual[:, 0], y_pred[:, 0]),
        'RMSE_Temperature': mean_squared_error(y_test_actual[:, 0], y_pred[:, 0]) ** 0.5,
        'R2_Temperature': r2_score(y_test_actual[:, 0], y_pred[:, 0]),
        'MAE_Humidity': mean_absolute_error(y_test_actual[:, 1], y_pred[:, 1]),
        'RMSE_Humidity': mean_squared_error(y_test_actual[:, 1], y_pred[:, 1]) ** 0.5,
        'R2_Humidity': r2_score(y_test_actual[:, 1], y_pred[:, 1]),
    }

    return model, metrics
