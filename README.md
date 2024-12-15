# Walmart Sales Forecasting Project

This project implements a comprehensive sales forecasting system for Walmart using historical sales data, incorporating holiday effects, and external features. The system uses both LSTM and XGBoost models for prediction, combining their results in an ensemble approach.

## Quick Start

1. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Place the following datasets in the `data/` directory:
   - train.csv
   - test.csv
   - features.csv
   - stores.csv

3. Run the complete pipeline using Jupyter Notebook:
   ```bash
   jupyter notebook main.ipynb
   ```

## Project Structure

```
walmart-sales-forecasting/
│
├── data/                     # Raw and processed data
│   ├── train.csv             # Raw training data
│   ├── test.csv              # Raw test data
│   ├── features.csv          # External features
│   ├── stores.csv            # Store metadata
│   ├── clean_*.csv           # Preprocessed data
│   ├── *_forecast.csv        # Model predictions
│   └── *_metrics.csv         # Model evaluation metrics
│
├── scripts/                  # Python scripts
│   ├── data_preprocessing.py # Data cleaning and feature engineering
│   ├── train_lstm.py         # LSTM model training
│   ├── train_xgboost.py      # XGBoost model training
│   ├── ensemble_results.py   # Ensemble predictions
│   └── export_powerbi.py     # Power BI data export
│
├── powerbi/                  # Power BI resources
│   ├── forecast_results.csv  # Final predictions
│   ├── store_metrics.csv     # Store-level metrics
│   ├── holiday_impact.csv    # Holiday analysis
│   ├── feature_importance.csv # Feature rankings
│   └── model_metrics.csv     # Model performance
│
├── models/                   # Saved model files
│   ├── lstm_model.h5         # Trained LSTM model
│   ├── xgboost_model.pkl     # Trained XGBoost model
│   └── sales_scaler.pkl      # Data scaler
│
├── main.ipynb               # Jupyter notebook to run pipeline
├── README.md                # Project documentation
└── requirements.txt         # Python dependencies
```

## Pipeline Components

### 1. Data Preprocessing
- Handles missing values using median/mode imputation
- Creates time-based features (Week, Month, Quarter)
- Implements lag features and rolling averages
- Generates holiday-related features
- Creates store type interaction features

### 2. LSTM Model
- Sequence-based deep learning model
- Two LSTM layers with dropout
- Early stopping to prevent overfitting
- Normalized sales data using MinMaxScaler

### 3. XGBoost Model
- Gradient boosting model
- Feature importance analysis
- Hyperparameter tuning
- Handles both numerical and categorical features

### 4. Ensemble Predictions
- Weighted average of LSTM and XGBoost predictions
- Weights based on model performance (MSE)
- Comprehensive evaluation metrics

### 5. Power BI Export
- Prepares data for visualization
- Creates store-level metrics
- Analyzes holiday impact
- Exports feature importance rankings

## Running the Pipeline

### Option 1: Using Jupyter Notebook (Recommended)
1. Open `main.ipynb`
2. Run all cells sequentially
3. Monitor progress and view intermediate results

### Option 2: Running Individual Scripts
```bash
cd scripts
python data_preprocessing.py
python train_lstm.py
python train_xgboost.py
python ensemble_results.py
python export_powerbi.py
```

## Evaluation Metrics
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R² Score
- Weighted Mean Absolute Error (WMAE)
  - Higher weights for holiday weeks

## Power BI Dashboard
The exported files in the `powerbi/` directory can be used to create visualizations:
- Time series plots of actual vs predicted sales
- Store-level performance analysis
- Holiday impact analysis
- Feature importance visualization
- Model performance metrics

## Requirements
- Python 3.8+
- TensorFlow 2.6+
- XGBoost 1.4+
- Pandas, NumPy, Scikit-learn
- Jupyter Notebook
- Power BI Desktop (for visualization)

## Notes
- Ensure consistent data types during merges
- Monitor memory usage during LSTM training
- Check for data leakage in feature engineering
- Review feature importance for model insights
