import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(filepath):
    df = pd.read_csv(filepath)

    # Combine 'Date' and 'Time' into one datetime column
    df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)

    # Drop original Date and Time columns
    df.drop(columns=['Date', 'Time'], inplace=True)

    # Sort and drop duplicates
    df.sort_values('timestamp', inplace=True)
    df.drop_duplicates(subset='timestamp', inplace=True)

    return df

def clean_data(df):
    # Only apply imputation to numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    imputer = SimpleImputer(strategy='mean')
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    return df

def scale_features(df, target_cols):
    # Drop timestamp and targets
    features = df.drop(columns=['timestamp'] + target_cols)

    # Keep only numeric features
    numeric_X = features.select_dtypes(include=['number'])

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_X)

    return pd.DataFrame(X_scaled, columns=numeric_X.columns), df[target_cols]


def prepare_data(filepath, test_size=0.2):
    df = load_data(filepath)
    df = clean_data(df)

    # Rename target columns for consistency with Part 2 design
    df.rename(columns={
        'Indoor_temperature_room': 'indoor_temperature',
        'Humidity': 'indoor_humidity'
    }, inplace=True)

    # Ensure targets exist
    if 'indoor_temperature' not in df.columns or 'indoor_humidity' not in df.columns:
        raise KeyError("Missing target columns after renaming.")

    target_cols = ['indoor_temperature', 'indoor_humidity']
    X, y = scale_features(df, target_cols)

    return train_test_split(X, y, test_size=test_size, random_state=42), df
