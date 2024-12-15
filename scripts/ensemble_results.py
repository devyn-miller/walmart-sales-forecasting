import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def load_predictions():
    """Load predictions from both models"""
    lstm_pred = pd.read_csv('data/lstm_forecast.csv')
    xgb_pred = pd.read_csv('data/xgboost_forecast.csv')
    return lstm_pred, xgb_pred

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
    lstm_pred, xgb_pred = load_predictions()
    lstm_metrics, xgb_metrics = load_metrics()
    
    # Calculate weights
    lstm_weight, xgb_weight = calculate_weights(lstm_metrics, xgb_metrics)
    print(f"Model weights - LSTM: {lstm_weight:.3f}, XGBoost: {xgb_weight:.3f}")
    
    # Create ensemble predictions
    ensemble_pred = (
        lstm_weight * lstm_pred['Predicted'] +
        xgb_weight * xgb_pred['Predicted']
    )
    
    # Evaluate ensemble
    metrics = evaluate_ensemble(lstm_pred['Actual'], ensemble_pred)
    
    # Save ensemble results
    results_df = pd.DataFrame({
        'Actual': lstm_pred['Actual'],
        'LSTM_Prediction': lstm_pred['Predicted'],
        'XGBoost_Prediction': xgb_pred['Predicted'],
        'Ensemble_Prediction': ensemble_pred
    })
    
    results_df.to_csv('data/final_forecast.csv', index=False)
    
    # Save ensemble metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('data/ensemble_metrics.csv', index=False)
    
    print("\nEnsemble predictions completed!")
    print("Metrics:", metrics)
    
    # Compare with individual model performance
    print("\nModel Performance Comparison:")
    print("LSTM Metrics:")
    print(lstm_metrics.to_dict('records')[0])
    print("\nXGBoost Metrics:")
    print(xgb_metrics.to_dict('records')[0])
    print("\nEnsemble Metrics:")
    print(metrics)

if __name__ == "__main__":
    main()
