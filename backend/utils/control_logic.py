# backend/utils/control_logic.py
import pandas as pd

def apply_controls(df: pd.DataFrame, output_path: str):
    if 'timestamp' not in df.columns:
        df['timestamp'] = pd.date_range(start='2012-03-13 11:45', periods=len(df), freq='15min')

    df['final_temperature'] = df['Pred_Temperature']
    df['final_humidity'] = df['Pred_Humidity']
    df['heater_status'] = 0
    df['ventilation_status'] = 0

    heater_limit_per_day = 2
    heater_duration = 16

    for i in range(len(df)):
        current_time = pd.to_datetime(df.loc[i, 'timestamp'])
        day_mask = df['timestamp'].apply(lambda t: pd.to_datetime(t).date() == current_time.date())
        heater_today = df.loc[day_mask, 'heater_status'].sum() // heater_duration

        temp = df.loc[i, 'Pred_Temperature']
        hum = df.loc[i, 'Pred_Humidity']
        out_temp = df.get('Outside temp', temp)
        out_hum = df.get('Outdoor_relative_humidity_Sensor', hum)

        if hum > 65:
            df.loc[i, 'final_temperature'] = out_temp
            df.loc[i, 'final_humidity'] = out_hum
            df.loc[i, 'ventilation_status'] = 1
            continue

        if temp < 20 and heater_today < heater_limit_per_day:
            for j in range(i, min(i + heater_duration, len(df))):
                if df.loc[j, 'ventilation_status'] == 1:
                    continue
                df.loc[j, 'final_temperature'] += 0.5
                df.loc[j, 'heater_status'] = 1
            continue

    df.to_csv(output_path, index=False)
