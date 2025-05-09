import pandas as pd

def apply_controls(df: pd.DataFrame, output_path: str,
                   temp_threshold=20,
                   humidity_threshold=59.5,
                   outdoor_temp_col='Outside temp',
                   outdoor_humidity_col='Outdoor_relative_humidity_Sensor',
                   heater_limit_per_day=2,
                   heater_duration=16,
                   ventilation_duration=8,
                   ventilation_cooldown=8,
                   ventilation_blend_alpha=0.4):
    """
    Applies optimized rule-based control logic on predicted temperature and humidity.
    """

    if 'timestamp' not in df.columns:
        df['timestamp'] = pd.date_range(start='2012-03-13 11:45', periods=len(df), freq='15min')

    df['final_temperature'] = df['Pred_Temperature']
    df['final_humidity'] = df['Pred_Humidity']
    df['heater_status'] = 0
    df['ventilation_status'] = 0

    daily_heater_uses = {}
    last_vent_index = -ventilation_cooldown * 2  # safe start

    for i in range(len(df)):
        timestamp = pd.to_datetime(df.loc[i, 'timestamp'])
        date = timestamp.date()

        temp = df.loc[i, 'Pred_Temperature']
        hum = df.loc[i, 'Pred_Humidity']
        out_temp = df[outdoor_temp_col].iloc[i] if outdoor_temp_col in df.columns and pd.notna(df[outdoor_temp_col].iloc[i]) else temp
        out_hum = df[outdoor_humidity_col].iloc[i] if outdoor_humidity_col in df.columns and pd.notna(df[outdoor_humidity_col].iloc[i]) else hum

        if date not in daily_heater_uses:
            daily_heater_uses[date] = 0

        print(f"[{timestamp}] Humidity={hum:.2f}, Temp={temp:.2f}, ΔVent={i - last_vent_index}")

        # VENTILATION LOGIC (blended, smarter)
        if hum >= humidity_threshold and (i - last_vent_index) >= ventilation_cooldown and out_hum < 65:
            for j in range(i, min(i + ventilation_duration, len(df))):
                df.loc[j, 'final_temperature'] = ventilation_blend_alpha * out_temp + (1 - ventilation_blend_alpha) * df.loc[j, 'final_temperature']
                df.loc[j, 'final_humidity'] = ventilation_blend_alpha * out_hum + (1 - ventilation_blend_alpha) * df.loc[j, 'final_humidity']
                df.loc[j, 'ventilation_status'] = 1
            last_vent_index = i
            print(f"🌬️ Ventilation triggered at {timestamp}")
            continue  # Skip heating if ventilating

        # HEATER LOGIC
        if temp < temp_threshold and daily_heater_uses[date] < heater_limit_per_day:
            can_heat = True
            for j in range(i, min(i + heater_duration, len(df))):
                if df.loc[j, 'ventilation_status'] == 1:
                    can_heat = False
                    break
            if can_heat:
                for j in range(i, min(i + heater_duration, len(df))):
                    df.loc[j, 'final_temperature'] += 0.5
                    df.loc[j, 'heater_status'] = 1
                daily_heater_uses[date] += 1
                print(f"🔥 Heater ON at {timestamp} (#{daily_heater_uses[date]} for {date})")

    df['heater_status'] = df['heater_status'].astype(int)
    df['ventilation_status'] = df['ventilation_status'].astype(int)
    df.to_csv(output_path, index=False)

    print(f"\n✅ Control-adjusted output saved to: {output_path}")
    print(f"🔁 Total ventilation activations: {df['ventilation_status'].sum()}")
    return df
