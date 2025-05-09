import pandas as pd
import matplotlib.pyplot as plt

def calculate_comfort_score(row, temp_range=(21, 23), hum_range=(40, 60)):
    temp = row.get('final_temperature', row.get('Pred_Temperature', 0))
    hum = row.get('final_humidity', row.get('Pred_Humidity', 0))
    score = 0

    # Temperature score (max 0.5)
    if temp_range[0] <= temp <= temp_range[1]:
        score += 0.5
    else:
        score += max(0, 0.5 - abs(temp - sum(temp_range) / 2) * 0.05)

    # Humidity score (max 0.5)
    if hum_range[0] <= hum <= hum_range[1]:
        score += 0.5
    else:
        score += max(0, 0.5 - abs(hum - sum(hum_range) / 2) * 0.01)

    return round(score, 3)

def analyze_comfort(csv_path, output_plot_path="output/adjusted_predictions_comfort_plot.png"):
    df = pd.read_csv(csv_path)
    df['comfort_score'] = df.apply(calculate_comfort_score, axis=1)

    avg_score = round(df['comfort_score'].mean(), 3)

    plt.figure(figsize=(12, 4))
    plt.plot(df['comfort_score'], label='Comfort Score', color='green')
    plt.axhline(y=1.0, color='blue', linestyle='--', label='Perfect Comfort')
    plt.axhline(y=0.7, color='orange', linestyle='--', label='Acceptable Threshold')
    plt.title("Comfort Score Over Time")
    plt.xlabel("Time")
    plt.ylabel("Comfort Score (0–1)")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_plot_path)
    plt.close()

    return avg_score, output_plot_path
