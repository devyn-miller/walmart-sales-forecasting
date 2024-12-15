# Walmart Sales Forecasting Project

This project implements a sales forecasting system for Walmart using historical sales data, incorporating holiday effects and external features. The system uses both LSTM and XGBoost models for prediction, combining their results in an ensemble approach.

## Project Structure

```
walmart-sales-forecasting/
│
├── data/                     # Raw and processed data
├── scripts/                  # Python scripts
├── powerbi/                  # Power BI dashboard resources
├── README.md                 # Project documentation
└── requirements.txt          # Python dependencies
```

## Setup Instructions

1. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Place the following datasets in the `data/` directory:
   - train.csv
   - test.csv
   - features.csv
   - stores.csv

3. Run the preprocessing script:
   ```bash
   cd scripts
   python data_preprocessing.py
   ```

## Pipeline Components

1. **Data Preprocessing** (`data_preprocessing.py`):
   - Loads and merges all datasets
   - Handles missing values
   - Creates time-based features
   - Implements lag features and rolling averages
   - Generates holiday-related features
   - Creates interaction features

2. **LSTM Model** (`train_lstm.py`) - Coming soon
3. **XGBoost Model** (`train_xgboost.py`) - Coming soon
4. **Ensemble Results** (`ensemble_results.py`) - Coming soon
5. **Power BI Export** (`export_powerbi.py`) - Coming soon

## Evaluation Metrics

The project uses the following metrics for model evaluation:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R² Score
- Weighted Mean Absolute Error (WMAE)

## Data Processing Details

### Feature Engineering
- **Time Features**: Week, Month, Quarter, Day of week
- **Lag Features**: 1-week and 4-week lagged sales
- **Rolling Features**: 4-week and 8-week moving averages
- **Holiday Features**: Binary flags and weeks to next holiday
- **Interaction Features**: Store type × Month, Markdown effects

### Missing Value Handling
- Numeric columns: Median imputation
- Categorical columns: Mode imputation

## Next Steps

1. Implement the LSTM model training script
2. Implement the XGBoost model training script
3. Create the ensemble combination script
4. Develop the Power BI dashboard
5. Add model evaluation and analysis
