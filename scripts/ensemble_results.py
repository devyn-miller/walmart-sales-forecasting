import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_predictions():
    """Load predictions from both models and ensure they are aligned"""
    lstm_pred = pd.read_csv('data/lstm_forecast.csv')
    xgb_pred = pd.read_csv('data/xgboost_forecast.csv')
    
    # Add index column if not present
    if 'index' not in lstm_pred.columns:
        lstm_pred = lstm_pred.reset_index()
    if 'index' not in xgb_pred.columns:
        xgb_pred = xgb_pred.reset_index()
    
    # Inner join to keep only matching rows
    merged_pred = pd.merge(lstm_pred, xgb_pred, 
                          on='index', 
                          suffixes=('_lstm', '_xgb'))
    
    # Verify actual values match
    actual_diff = (merged_pred['Actual_lstm'] - merged_pred['Actual_xgb']).abs().mean()
    if actual_diff > 1e-6:
        print("Warning: Actual values differ between LSTM and XGBoost predictions")
        print(f"Mean absolute difference: {actual_diff}")
    
    # Use LSTM actuals as ground truth
    predictions = {
        'Actual': merged_pred['Actual_lstm'],
        'LSTM_Pred': merged_pred['Predicted_lstm'],
        'XGB_Pred': merged_pred['Predicted_xgb']
    }
    
    return pd.DataFrame(predictions)

def load_metrics():
    """Load metrics from both models"""
    lstm_metrics = pd.read_csv('data/lstm_metrics.csv')
    xgb_metrics = pd.read_csv('data/xgboost_metrics.csv')
    return lstm_metrics, xgb_metrics

def calculate_weights(lstm_metrics, xgb_metrics):
    """Calculate weights based on model performance (using MSE)"""
    lstm_mse = lstm_metrics['MSE'].values[0]
    xgb_mse = xgb_metrics['MSE'].values[0]
    
    # Convert MSE to weights (lower MSE = higher weight)
    total = (1/lstm_mse) + (1/xgb_mse)
    lstm_weight = (1/lstm_mse) / total
    xgb_weight = (1/xgb_mse) / total
    
    return lstm_weight, xgb_weight

def evaluate_ensemble(y_true, y_pred):
    """Calculate evaluation metrics for ensemble predictions"""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate WMAE
    holiday_weights = np.where(y_true > y_true.mean(), 1.5, 1.0)
    wmae = np.average(np.abs(y_true - y_pred), weights=holiday_weights)
    
    return {
        'MSE': mse,
        'MAE': mae,
        'R2': r2,
        'WMAE': wmae
    }

def main():
    print("Loading predictions and metrics...")
    predictions = load_predictions()
    lstm_metrics, xgb_metrics = load_metrics()
    
    # Calculate weights
    lstm_weight, xgb_weight = calculate_weights(lstm_metrics, xgb_metrics)
    print(f"Model weights - LSTM: {lstm_weight:.3f}, XGBoost: {xgb_weight:.3f}")
    
    # Create ensemble predictions
    ensemble_pred = (
        lstm_weight * predictions['LSTM_Pred'] +
        xgb_weight * predictions['XGB_Pred']
    )
    
    # Evaluate ensemble
    metrics = evaluate_ensemble(predictions['Actual'], ensemble_pred)
    print("\nModel Performance Comparison:")
    
    # Calculate individual model metrics
    lstm_metrics_new = evaluate_ensemble(predictions['Actual'], predictions['LSTM_Pred'])
    xgb_metrics_new = evaluate_ensemble(predictions['Actual'], predictions['XGB_Pred'])
    
    print("\nLSTM Metrics:")
    print(lstm_metrics_new)
    print("\nXGBoost Metrics:")
    print(xgb_metrics_new)
    print("\nEnsemble Metrics:")
    print(metrics)
    
    # Save ensemble results
    results_df = pd.DataFrame({
        'Actual': predictions['Actual'],
        'LSTM_Prediction': predictions['LSTM_Pred'],
        'XGBoost_Prediction': predictions['XGB_Pred'],
        'Ensemble_Prediction': ensemble_pred
    })
    
    results_df.to_csv('data/final_forecast.csv', index=False)
    
    # Save ensemble metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('data/ensemble_metrics.csv', index=False)
    
    print("\nEnsemble predictions completed!")

if __name__ == "__main__":
    main()
