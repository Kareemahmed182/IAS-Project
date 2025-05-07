import pandas as pd

def apply_controls(predictions_path, output_path,
                   outdoor_temp_col='Outside temp',
                   outdoor_humidity_col='Outdoor_relative_humidity_Sensor',
                   temp_threshold=20, humidity_threshold=65):
    """
    Apply adaptive control logic (heater & ventilation) to predicted indoor conditions.
    """

    df = pd.read_csv(predictions_path)

    # Ensure timestamp exists
    if 'timestamp' not in df.columns:
        df['timestamp'] = pd.date_range(start='2012-03-13 11:45', periods=len(df), freq='15min')

    # Initialize final outputs
    df['final_temperature'] = df['Pred_Temperature']
    df['final_humidity'] = df['Pred_Humidity']
    df['heater_status'] = 0
    df['ventilation_status'] = 0

    # Control constraints
    heater_limit_per_day = 2
    heater_duration = 16  # = 4 hours

    for i in range(len(df)):
        current_time = pd.to_datetime(df.loc[i, 'timestamp'])
        today = current_time.date()
        day_mask = pd.to_datetime(df['timestamp']).dt.date == today
        heater_today_count = df.loc[day_mask, 'heater_status'].sum() // heater_duration

        indoor_temp = df.loc[i, 'Pred_Temperature']
        indoor_hum = df.loc[i, 'Pred_Humidity']

        outdoor_temp = df.loc[i, outdoor_temp_col] if outdoor_temp_col in df.columns else indoor_temp
        outdoor_hum = df.loc[i, outdoor_humidity_col] if outdoor_humidity_col in df.columns else indoor_hum

        # VENTILATION RULE
        if indoor_hum > humidity_threshold:
            df.loc[i, 'final_temperature'] = outdoor_temp
            df.loc[i, 'final_humidity'] = outdoor_hum
            df.loc[i, 'ventilation_status'] = 1
            continue  # skip heater if ventilating

        # HEATER RULE
        if indoor_temp < temp_threshold and heater_today_count < heater_limit_per_day:
            for j in range(i, min(i + heater_duration, len(df))):
                if df.loc[j, 'ventilation_status'] == 1:
                    continue
                df.loc[j, 'final_temperature'] += 0.5
                df.loc[j, 'heater_status'] = 1
            continue

    # Ensure numeric output for Streamlit plotting
    df['heater_status'] = df['heater_status'].astype(int)
    df['ventilation_status'] = df['ventilation_status'].astype(int)

    df.to_csv(output_path, index=False)
    print(f"✅ Control-adjusted output saved to: {output_path}")
    return df
