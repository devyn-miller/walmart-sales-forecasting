import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os

def load_data():
    """Load the preprocessed training data"""
    train_data = pd.read_csv('../data/clean_train_data.csv')
    train_data['Date'] = pd.to_datetime(train_data['Date'])
    return train_data

def prepare_sequences(data, sequence_length=4):
    """Prepare sequences for LSTM model"""
    features = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
                'is_holiday', 'Week', 'Month', 'Sales_Rolling4', 'Sales_Rolling8']
    
    X, y = [], []
    for store in data['Store'].unique():
        store_data = data[data['Store'] == store][features].values
        for i in range(len(store_data) - sequence_length):
            X.append(store_data[i:(i + sequence_length)])
            y.append(store_data[i + sequence_length, 0])  # Weekly_Sales is at index 0
    
    return np.array(X), np.array(y)

def create_lstm_model(input_shape):
    """Create and compile LSTM model"""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def evaluate_model(y_true, y_pred):
    """Calculate evaluation metrics"""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate WMAE (Weighted MAE with higher weights for holiday weeks)
    holiday_weights = np.where(y_true > y_true.mean(), 1.5, 1.0)  # Higher weights for above-average sales
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
    
    # Create scalers directory if it doesn't exist
    os.makedirs('../models', exist_ok=True)
    
    # Scale the features
    scaler = MinMaxScaler()
    data['Weekly_Sales'] = scaler.fit_transform(data[['Weekly_Sales']])
    joblib.dump(scaler, '../models/sales_scaler.pkl')
    
    # Prepare sequences
    X, y = prepare_sequences(data)
    
    # Split into train and validation sets
    train_split = int(0.8 * len(X))
    X_train, X_val = X[:train_split], X[train_split:]
    y_train, y_val = y[:train_split], y[train_split:]
    
    print("Training LSTM model...")
    model = create_lstm_model(input_shape=(X.shape[1], X.shape[2]))
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Save the model
    model.save('../models/lstm_model.h5')
    
    # Make predictions
    y_pred = model.predict(X_val)
    
    # Inverse transform predictions and actual values
    y_pred = scaler.inverse_transform(y_pred)
    y_val = scaler.inverse_transform(y_val.reshape(-1, 1))
    
    # Evaluate model
    metrics = evaluate_model(y_val, y_pred)
    
    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('../data/lstm_metrics.csv', index=False)
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'Actual': y_val.flatten(),
        'Predicted': y_pred.flatten()
    })
    predictions_df.to_csv('../data/lstm_forecast.csv', index=False)
    
    print("LSTM Training completed!")
    print("Metrics:", metrics)

if __name__ == "__main__":
    main()
