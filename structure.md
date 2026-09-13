Customer_Churn_Prediction/
│
├── README.md                 # Project overview + workflow + results
├── churn1.ipynb              # Model training & analysis
├── predict.py                # Interactive prediction
├── requirements.txt          # Required libraries
│
├── data/
│   └── telco_churn.csv       # Dataset
│
└── models/
    ├── lr_model.pkl          # Logistic Regression
    ├── rf_model.pkl          # Random Forest
    ├── xgb_model.pkl         # XGBoost
    ├── catboost_model.pkl    # CatBoost
    ├── scaler.pkl            # Numerical scaler
    ├── feature_columns.pkl   # OHE feature columns
    └── ann_model.keras       # ANN model
