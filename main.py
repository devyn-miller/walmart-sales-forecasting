# %% [markdown]
# # Walmart Sales Forecasting Pipeline
# 
# This notebook runs the complete sales forecasting pipeline, including:
# 1. Data preprocessing
# 2. LSTM model training
# 3. XGBoost model training
# 4. Ensemble predictions
# 5. Power BI data export
# 
# Make sure you have all required datasets in the `data/` directory before running this notebook.

# %%
import os
import sys
import pandas as pd
import numpy as np
import argparse
from IPython.display import display, HTML

# Add scripts directory to path
sys.path.append('scripts')

# Import our modules
import data_preprocessing
import train_lstm
import train_xgboost
import ensemble_results
import export_powerbi

def check_required_files():
    """Check if all required input files are present"""
    required_files = ['train.csv', 'test.csv', 'features.csv', 'stores.csv']
    missing_files = []

    for file in required_files:
        if not os.path.exists(f'data/{file}'):
            missing_files.append(file)

    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease add these files to the 'data/' directory before proceeding.")
        return False
    print("✅ All required files present!")
    return True

def run_preprocessing(force=False):
    """Run data preprocessing if needed"""
    if not force and os.path.exists('data/clean_train_data.csv'):
        print("✅ Using existing preprocessed data")
        return True
    
    print("Starting data preprocessing...")
    data_preprocessing.main()
    print("\nPreprocessing completed!")
    
    # Display sample of processed data
    clean_train = pd.read_csv('data/clean_train_data.csv')
    display(HTML(clean_train.head().to_html()))
    return True

def run_lstm(force=False):
    """Run LSTM training if needed"""
    if not force and os.path.exists('data/lstm_metrics.csv') and os.path.exists('data/lstm_forecast.csv'):
        print("✅ Using existing LSTM model and predictions")
        return True
    
    print("Training LSTM model...")
    train_lstm.main()

    # Display LSTM metrics
    lstm_metrics = pd.read_csv('data/lstm_metrics.csv')
    display(HTML(lstm_metrics.to_html()))
    return True

def run_xgboost(force=False):
    """Run XGBoost training if needed"""
    if not force and os.path.exists('data/xgboost_metrics.csv') and os.path.exists('data/xgboost_forecast.csv'):
        print("✅ Using existing XGBoost model and predictions")
        return True
    
    print("Training XGBoost model...")
    train_xgboost.main()

    # Display metrics and feature importance
    xgb_metrics = pd.read_csv('data/xgboost_metrics.csv')
    feature_importance = pd.read_csv('data/feature_importance.csv')

    print("\nXGBoost Metrics:")
    display(HTML(xgb_metrics.to_html()))

    print("\nTop 10 Important Features:")
    display(HTML(feature_importance.head(10).to_html()))
    return True

def run_ensemble(force=False):
    """Run ensemble predictions if needed"""
    if not force and os.path.exists('data/ensemble_metrics.csv'):
        print("✅ Using existing ensemble predictions")
        return True
    
    print("Creating ensemble predictions...")
    ensemble_results.main()

    # Display ensemble metrics
    if os.path.exists('data/ensemble_metrics.csv'):
        ensemble_metrics = pd.read_csv('data/ensemble_metrics.csv')
        display(HTML(ensemble_metrics.to_html()))
    return True

def run_powerbi_export(force=False):
    """Run Power BI export if needed"""
    if not force and os.path.exists('data/powerbi_export.csv'):
        print("✅ Using existing Power BI export")
        return True
    
    print("Exporting data for Power BI...")
    export_powerbi.main()
    return True

def main():
    parser = argparse.ArgumentParser(description='Run Walmart Sales Forecasting Pipeline')
    parser.add_argument('--start-from', choices=['preprocess', 'lstm', 'xgboost', 'ensemble', 'powerbi'], 
                      help='Start pipeline from specific step')
    parser.add_argument('--force', action='store_true', 
                      help='Force rerun of all steps from start-from onwards')
    args = parser.parse_args()

    # Define pipeline steps
    pipeline_steps = {
        'preprocess': run_preprocessing,
        'lstm': run_lstm,
        'xgboost': run_xgboost,
        'ensemble': run_ensemble,
        'powerbi': run_powerbi_export
    }

    # Check required files
    if not check_required_files():
        return

    # Determine where to start
    start_step = args.start_from if args.start_from else 'preprocess'
    started = False

    # Run pipeline steps
    for step_name, step_func in pipeline_steps.items():
        if step_name == start_step:
            started = True
        if started:
            if not step_func(force=args.force):
                print(f"❌ Failed at step: {step_name}")
                return
            
    print("✅ Pipeline completed successfully!")

if __name__ == "__main__":
    main()
