from backend.utils.preprocessing import prepare_data
from backend.models.lstm_model import train_and_evaluate_lstm
from backend.models.rf_model import train_and_evaluate_rf
from backend.utils.plot_utils import plot_predictions, plot_feature_importance
from backend.utils.plot_utils import plot_controls_over_time
from control_simulation import apply_controls

# Step 1: Load and prepare data
(X_train, X_test, y_train, y_test), df = prepare_data("data/6FTC2088.csv")

print("✅ Data Loaded")
print("Training samples:", len(X_train))
print("Features:", X_train.shape[1])
print("Targets:", y_train.columns.tolist())
print("First timestamp:", df['timestamp'].iloc[0])

# Step 2: Train and evaluate LSTM
model_lstm, lstm_metrics = train_and_evaluate_lstm(X_train, y_train, X_test, y_test ,timesteps=20)

print("\n📊 LSTM Evaluation Metrics:")
for k, v in lstm_metrics.items():
    print(f"{k}: {v:.3f}")

plot_predictions("output/predictions_lstm.csv", "LSTM")

# Step 3: Train and evaluate Random Forest
model_rf, rf_metrics = train_and_evaluate_rf(X_train, y_train, X_test, y_test)

print("\n🌲 Random Forest Evaluation Metrics:")
for k, v in rf_metrics.items():
    print(f"{k}: {v:.3f}")

plot_predictions("output/predictions_rf.csv", "Random Forest")
plot_feature_importance(model_rf, X_train.columns, "Random Forest")

# Step 4: Simulate adaptive control (ventilation + heater)
adjusted_df = apply_controls(
    "output/predictions_rf.csv",
    "output/adjusted_predictions.csv"
)

print("\n✅ Control-adjusted predictions saved.")

# Step 5: Plot showing how heater and ventilation status changed over time in response to control logic
plot_controls_over_time("output/adjusted_predictions.csv")

