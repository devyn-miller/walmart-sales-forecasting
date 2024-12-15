import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import joblib
import os

def load_data():
    """Load the preprocessed training data"""
    train_data = pd.read_csv('data/clean_train_data.csv')
    train_data['Date'] = pd.to_datetime(train_data['Date'])
    return train_data

def prepare_features(data):
    """Prepare features for XGBoost model"""
    # Select relevant features
    features = [
        'Store', 'Dept', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
        'is_holiday', 'Week', 'Month', 'Quarter', 'Year',
        'Sales_Rolling4', 'Sales_Rolling8', 'weeks_to_holiday'
    ]
    
    # Add markdown features if they exist
    markdown_cols = [col for col in data.columns if 'MarkDown' in col]
    features.extend(markdown_cols)
    
    # Encode categorical variables
    le = LabelEncoder()
    if 'Type' in data.columns:
        data['Type_encoded'] = le.fit_transform(data['Type'])
        features.append('Type_encoded')
    
    X = data[features]
    y = data['Weekly_Sales']
    
    return X, y

def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost model with hyperparameter tuning"""
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 1000,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'early_stopping_rounds': 10
    }
    
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )
    
    return model

def evaluate_model(y_true, y_pred):
    """Calculate evaluation metrics"""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate WMAE (Weighted MAE with higher weights for holiday weeks)
    holiday_weights = np.where(y_true > y_true.mean(), 1.5, 1.0)
    wmae = np.average(np.abs(y_true - y_pred), weights=holiday_weights)
    
    return {
        'MSE': mse,
        'MAE': mae,
        'R2': r2,
        'WMAE': wmae
    }

def main():
    print("Loading and preparing data...")
    data = load_data()
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Load the same scaler used by LSTM
    try:
        scaler = joblib.load('models/sales_scaler.pkl')
        print("Using existing sales scaler from LSTM")
    except:
        print("Creating new sales scaler")
        scaler = MinMaxScaler()
        scaler.fit(data[['Weekly_Sales']])
        joblib.dump(scaler, 'models/sales_scaler.pkl')
    
    # Scale the target variable
    data['Weekly_Sales'] = scaler.transform(data[['Weekly_Sales']])
    
    # Prepare features
    X, y = prepare_features(data)
    
    # Use temporal split (same as LSTM)
    train_split = int(0.8 * len(X))
    X_train, X_val = X.iloc[:train_split], X.iloc[train_split:]
    y_train, y_val = y.iloc[:train_split], y.iloc[train_split:]
    
    print("Training XGBoost model...")
    model = train_xgboost(X_train, y_train, X_val, y_val)
    
    # Save the model
    joblib.dump(model, 'models/xgboost_model.pkl')
    
    # Make predictions
    y_pred = model.predict(X_val)
    
    # Inverse transform predictions and actual values
    y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_val = scaler.inverse_transform(y_val.values.reshape(-1, 1))
    
    # Evaluate model
    metrics = evaluate_model(y_val, y_pred)
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('data/xgboost_metrics.csv', index=False)
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'Actual': y_val.flatten(),
        'Predicted': y_pred.flatten()
    })
    predictions_df.to_csv('data/xgboost_forecast.csv', index=False)
    
    # Save feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    feature_importance.to_csv('data/feature_importance.csv', index=False)
    
    print("XGBoost Training completed!")
    print("Metrics:", metrics)
    print("\nTop 10 Important Features:")
    print(feature_importance.head(10))

if __name__ == "__main__":
    main()
