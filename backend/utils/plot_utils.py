import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_predictions(csv_path, model_name):
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(12, 5))

    # Plot for temperature
    plt.subplot(1, 2, 1)
    plt.plot(df['Actual_Temperature'], label='Actual', linewidth=2)
    plt.plot(df['Pred_Temperature'], label='Predicted', linestyle='--')
    plt.title(f'{model_name} - Indoor Temperature')
    plt.xlabel('Sample')
    plt.ylabel('°C')
    plt.legend()

    # Plot for humidity
    plt.subplot(1, 2, 2)
    plt.plot(df['Actual_Humidity'], label='Actual', linewidth=2)
    plt.plot(df['Pred_Humidity'], label='Predicted', linestyle='--')
    plt.title(f'{model_name} - Indoor Humidity')
    plt.xlabel('Sample')
    plt.ylabel('%')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'output/{model_name}_predictions_plot.png', dpi=300)
    plt.show()



def plot_feature_importance(model, feature_names, model_name="Random Forest"):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title(f"{model_name} - Feature Importance")
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.ylabel('Importance Score')
    plt.tight_layout()
    plt.savefig(f'output/{model_name}_feature_importance.png', dpi=300)
    plt.show()

def plot_controls_over_time(filepath, output_path="output/control_plot.png"):
    df = pd.read_csv(filepath)

    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    plt.figure(figsize=(14, 6))

    plt.plot(df['timestamp'], df['heater_status'], label='Heater Status', drawstyle='steps-post')
    plt.plot(df['timestamp'], df['ventilation_status'], label='Ventilation Status', drawstyle='steps-post')

    plt.xlabel('Time')
    plt.ylabel('Status')
    plt.title('Heater and Ventilation Status Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"✅ Control plot saved to: {output_path}")
    plt.show()

