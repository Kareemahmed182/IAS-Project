import os
import sys
import warnings
import logging
import pandas as pd

# ✅ Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all, 1 = info, 2 = warning, 3 = error
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger('tensorflow').setLevel(logging.FATAL)

from backend.utils.preprocessing import prepare_data
from backend.models.lstm_model import train_and_evaluate_lstm
from backend.models.rf_model import train_and_evaluate_rf
from backend.utils.plot_utils import plot_predictions, plot_feature_importance, plot_controls_over_time
from backend.utils.comfort_analysis import analyze_comfort
from backend.utils.fuzzy_controller import apply_fuzzy_controls
from control_simulation import apply_controls

# 📁 Create output folder
os.makedirs("output", exist_ok=True)

# 🧼 Delete only non-cleaned CSVs (keep cleaned + plots)
for fname in os.listdir("output"):
    if fname.endswith(".csv") and not fname.startswith("cleaned_"):
        os.remove(os.path.join("output", fname))

# STEP 1️⃣: Load and prepare dataset
(X_train, X_test, y_train, y_test), df = prepare_data("data/6FTC2088.csv")
print("✅ Data Loaded")
print("Training samples:", len(X_train))
print("Features:", X_train.shape[1])
print("Targets:", y_train.columns.tolist())
print("First timestamp:", df['timestamp'].iloc[0])

# STEP 2️⃣: Train and evaluate LSTM model
model_lstm, lstm_metrics = train_and_evaluate_lstm(X_train, y_train, X_test, y_test, timesteps=20)
print("\n📊 LSTM Evaluation Metrics:")
for k, v in lstm_metrics.items():
    print(f"{k}: {v:.3f}")
plot_predictions("output/predictions_lstm.csv", "LSTM")

# 📦 Clean and export LSTM predictions
lstm_df = pd.read_csv("output/predictions_lstm.csv")
lstm_df['timestamp'] = pd.date_range(start='2012-03-13 11:45', periods=len(lstm_df), freq='15min')
lstm_df[['timestamp', 'Pred_Temperature', 'Pred_Humidity', 'Actual_Temperature', 'Actual_Humidity']] \
    .to_csv("output/cleaned_predictions_lstm.csv", index=False)

# STEP 3️⃣: Train and evaluate Random Forest model
model_rf, rf_metrics = train_and_evaluate_rf(X_train, y_train, X_test, y_test)
print("\n🌲 Random Forest Evaluation Metrics:")
for k, v in rf_metrics.items():
    print(f"{k}: {v:.3f}")
plot_predictions("output/predictions_rf.csv", "Random Forest")
plot_feature_importance(model_rf, X_train.columns, "Random Forest")

# 📦 Clean and export RF predictions
rf_df = pd.read_csv("output/predictions_rf.csv")
rf_df['timestamp'] = pd.date_range(start='2012-03-13 11:45', periods=len(rf_df), freq='15min')
rf_df['Actual_Temperature'] = y_test.iloc[:, 0].values
rf_df['Actual_Humidity'] = y_test.iloc[:, 1].values
rf_df[['timestamp', 'Pred_Temperature', 'Pred_Humidity', 'Actual_Temperature', 'Actual_Humidity']] \
    .to_csv("output/cleaned_predictions_rf.csv", index=False)

# STEP 4️⃣: Apply rule-based control
adjusted_df = apply_controls(rf_df, "output/adjusted_predictions.csv")
plot_controls_over_time("output/adjusted_predictions.csv")

# 🧮 Add comfort scores and save cleaned version
adjusted_df['timestamp'] = pd.date_range(start='2012-03-13 11:45', periods=len(adjusted_df), freq='15min')
avg_score, _ = analyze_comfort("output/adjusted_predictions.csv", "output/adjusted_predictions_comfort_plot.png")
adjusted_df['comfort_score'] = adjusted_df.apply(lambda row: round(
    0.5 * (1 - min(abs(row['final_temperature'] - 22), 10) / 10) +
    0.5 * (1 - min(abs(row['final_humidity'] - 50), 20) / 20), 3), axis=1)

adjusted_df[['timestamp', 'Pred_Temperature', 'Pred_Humidity',
             'final_temperature', 'final_humidity',
             'heater_status', 'ventilation_status',
             'comfort_score']] \
    .to_csv("output/cleaned_adjusted_predictions.csv", index=False)

print(f"\n🎯 Final Comfort Score: {avg_score:.3f}")

# STEP 5️⃣: Fuzzy logic control
apply_fuzzy_controls("output/predictions_rf.csv", "output/fuzzy_decisions.csv")

# 📦 Clean and export fuzzy controller results
fuzzy_df = pd.read_csv("output/fuzzy_decisions.csv")
fuzzy_df['timestamp'] = pd.date_range(start='2012-03-13 11:45', periods=len(fuzzy_df), freq='15min')
fuzzy_df[['timestamp', 'Pred_Temperature', 'Pred_Humidity', 'fuzzy_heater', 'fuzzy_ventilation']] \
    .to_csv("output/cleaned_fuzzy_decisions.csv", index=False)
