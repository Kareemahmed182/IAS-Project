import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_data(filepath):
    df = pd.read_csv(filepath)

    # Combine Date and Time into datetime
    df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)

    # Drop raw Date and Time
    df.drop(columns=['Date', 'Time'], inplace=True)

    # Sort chronologically and remove duplicates
    df.sort_values('timestamp', inplace=True)
    df.drop_duplicates(subset='timestamp', inplace=True)

    return df

def clean_data(df):
    # Drop low-value or redundant features
    to_drop = [
        'Satisfaction', 'Relative_humidity_room',
        'Meteo_Sun_light_in_west_facade',
        'Occupancy 3', 'Meteo_Sun_light_in_north_facade'
    ]
    df = df.drop(columns=[col for col in to_drop if col in df.columns], errors='ignore')

    # Impute missing values for numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    imputer = SimpleImputer(strategy='mean')
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    return df

def scale_features(df, target_cols):
    # Separate inputs and targets
    X = df.drop(columns=['timestamp'] + target_cols)
    y = df[target_cols]

    # Scale numeric features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns), y

def prepare_data(filepath, test_size=0.2):
    df = load_data(filepath)
    df = clean_data(df)

    # Rename targets to match model expectations
    df.rename(columns={
        'Indoor_temperature_room': 'indoor_temperature',
        'Humidity': 'indoor_humidity'
    }, inplace=True)

    target_cols = ['indoor_temperature', 'indoor_humidity']
    if not all(col in df.columns for col in target_cols):
        raise KeyError("Missing one or both target columns.")

    # Scale features
    X, y = scale_features(df, target_cols)

    # Chronological split (not random)
    split_index = int(len(df) * (1 - test_size))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    return (X_train, X_test, y_train, y_test), df
