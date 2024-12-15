import pandas as pd
import numpy as np
from datetime import datetime

def load_data():
    """Load all necessary data for Power BI"""
    forecast_data = pd.read_csv('../data/final_forecast.csv')
    clean_train = pd.read_csv('../data/clean_train_data.csv')
    metrics = pd.read_csv('../data/ensemble_metrics.csv')
    feature_importance = pd.read_csv('../data/feature_importance.csv')
    return forecast_data, clean_train, metrics, feature_importance

def prepare_time_series_data(forecast_data, clean_train):
    """Prepare time series data for visualization"""
    # Add relevant columns from clean_train to forecast data
    forecast_with_metadata = pd.merge(
        forecast_data,
        clean_train[['Store', 'Dept', 'Date', 'Type', 'Size', 'is_holiday']],
        left_index=True,
        right_index=True
    )
    
    # Convert Date to datetime
    forecast_with_metadata['Date'] = pd.to_datetime(forecast_with_metadata['Date'])
    
    # Add time-based columns for better filtering in Power BI
    forecast_with_metadata['Year'] = forecast_with_metadata['Date'].dt.year
    forecast_with_metadata['Month'] = forecast_with_metadata['Date'].dt.month
    forecast_with_metadata['Week'] = forecast_with_metadata['Date'].dt.isocalendar().week
    
    return forecast_with_metadata

def calculate_store_metrics(data):
    """Calculate store-level metrics"""
    store_metrics = data.groupby('Store').agg({
        'Actual': ['mean', 'std', 'min', 'max'],
        'Ensemble_Prediction': ['mean', 'std', 'min', 'max'],
        'Type': 'first',
        'Size': 'first'
    }).reset_index()
    
    # Flatten column names
    store_metrics.columns = [
        'Store', 'Actual_Mean', 'Actual_Std', 'Actual_Min', 'Actual_Max',
        'Predicted_Mean', 'Predicted_Std', 'Predicted_Min', 'Predicted_Max',
        'Type', 'Size'
    ]
    
    return store_metrics

def calculate_holiday_impact(data):
    """Calculate holiday impact on sales"""
    holiday_impact = data.groupby('is_holiday').agg({
        'Actual': 'mean',
        'Ensemble_Prediction': 'mean'
    }).reset_index()
    
    holiday_impact['Sales_Lift'] = (
        holiday_impact['Actual'] / holiday_impact['Actual'].mean() - 1
    ) * 100
    
    return holiday_impact

def main():
    print("Loading and preparing data for Power BI...")
    forecast_data, clean_train, metrics, feature_importance = load_data()
    
    # Prepare main forecast dataset
    powerbi_data = prepare_time_series_data(forecast_data, clean_train)
    
    # Calculate additional metrics
    store_metrics = calculate_store_metrics(powerbi_data)
    holiday_impact = calculate_holiday_impact(powerbi_data)
    
    # Export datasets for Power BI
    print("Exporting data for Power BI...")
    
    # Main forecast results
    powerbi_data.to_csv('../powerbi/forecast_results.csv', index=False)
    
    # Store-level metrics
    store_metrics.to_csv('../powerbi/store_metrics.csv', index=False)
    
    # Holiday impact analysis
    holiday_impact.to_csv('../powerbi/holiday_impact.csv', index=False)
    
    # Feature importance
    feature_importance.to_csv('../powerbi/feature_importance.csv', index=False)
    
    # Model performance metrics
    metrics.to_csv('../powerbi/model_metrics.csv', index=False)
    
    print("Data export completed! Files are ready for Power BI visualization.")
    print("\nExported files:")
    print("1. forecast_results.csv - Main forecast data with metadata")
    print("2. store_metrics.csv - Store-level performance metrics")
    print("3. holiday_impact.csv - Holiday impact analysis")
    print("4. feature_importance.csv - Feature importance rankings")
    print("5. model_metrics.csv - Model performance metrics")

if __name__ == "__main__":
    main()
