import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load all required datasets"""
    print("Loading datasets...")
    train_df = pd.read_csv('../data/train.csv')
    test_df = pd.read_csv('../data/test.csv')
    features_df = pd.read_csv('../data/features.csv')
    stores_df = pd.read_csv('../data/stores.csv')
    
    return train_df, test_df, features_df, stores_df

def preprocess_dates(df):
    """Convert Date columns to datetime format"""
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def handle_missing_values(df):
    """Handle missing values in the dataset"""
    # Fill numeric columns with median
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].median())
    
    # Fill categorical columns with mode
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    return df

def create_time_features(df):
    """Create time-based features"""
    df['Week'] = df['Date'].dt.isocalendar().week
    df['Month'] = df['Date'].dt.month
    df['Quarter'] = df['Date'].dt.quarter
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Year'] = df['Date'].dt.year
    return df

def create_lag_features(df):
    """Create lagged sales features"""
    # Group by Store and Department
    grouped = df.groupby(['Store', 'Dept'])
    
    # Create 1-week and 4-week lag features
    df['Sales_Lag1'] = grouped['Weekly_Sales'].shift(1)
    df['Sales_Lag4'] = grouped['Weekly_Sales'].shift(4)
    
    # Create rolling averages
    df['Sales_Rolling4'] = grouped['Weekly_Sales'].transform(
        lambda x: x.rolling(window=4, min_periods=1).mean())
    df['Sales_Rolling8'] = grouped['Weekly_Sales'].transform(
        lambda x: x.rolling(window=8, min_periods=1).mean())
    
    return df

def create_holiday_features(df):
    """Create holiday-related features"""
    # Convert IsHoliday to numeric
    df['is_holiday'] = df['IsHoliday'].astype(int)
    
    # Create weeks to next holiday feature (simplified version)
    df['weeks_to_holiday'] = df.groupby(['Store', 'Dept'])['is_holiday'].transform(
        lambda x: x.rolling(window=52, min_periods=1).sum())
    
    return df

def create_interaction_features(df):
    """Create interaction features"""
    df['Store_Type_Month'] = df['Type'] + '_' + df['Month'].astype(str)
    
    # Create markdown interaction features
    markdown_cols = [col for col in df.columns if 'MarkDown' in col]
    for col in markdown_cols:
        df[f'{col}_by_type'] = df[col] * pd.factorize(df['Type'])[0]
    
    return df

def main():
    # Load data
    train_df, test_df, features_df, stores_df = load_data()
    
    # Preprocess dates
    train_df = preprocess_dates(train_df)
    test_df = preprocess_dates(test_df)
    features_df = preprocess_dates(features_df)
    
    # Merge datasets
    # First merge train/test with features
    train_merged = pd.merge(train_df, features_df, on=['Store', 'Date'], how='left')
    test_merged = pd.merge(test_df, features_df, on=['Store', 'Date'], how='left')
    
    # Then merge with stores data
    train_final = pd.merge(train_merged, stores_df, on='Store', how='left')
    test_final = pd.merge(test_merged, stores_df, on='Store', how='left')
    
    # Handle missing values
    train_final = handle_missing_values(train_final)
    test_final = handle_missing_values(test_final)
    
    # Create features
    for df in [train_final, test_final]:
        df = create_time_features(df)
        df = create_holiday_features(df)
        df = create_interaction_features(df)
    
    # Create lag features only for training data
    train_final = create_lag_features(train_final)
    
    # Save processed data
    print("Saving processed data...")
    train_final.to_csv('../data/clean_train_data.csv', index=False)
    test_final.to_csv('../data/clean_test_data.csv', index=False)
    print("Data preprocessing completed!")

if __name__ == "__main__":
    main()
