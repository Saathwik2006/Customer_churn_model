import pandas as pd
import joblib
from tensorflow.keras.models import load_model


# Load models
lr = joblib.load("models/lr_model.pkl")
rf = joblib.load("models/rf_model.pkl")
xgb = joblib.load("models/xgb_model.pkl")
cb = joblib.load("models/catboost_model.pkl")

# Load preprocessing
scaler = joblib.load("models/scaler.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

# Load ANN
ann = load_model("models/ann_model.keras")

def predict_churn():

    inputs = [
        input("Gender (Male/Female): "),
        int(input("Senior Citizen (0/1): ")),
        input("Partner (Yes/No): "),
        input("Dependents (Yes/No): "),
        int(input("Tenure (months): ")),
        input("Phone Service (Yes/No): "),
        input("Multiple Lines (Yes/No/No phone service): "),
        input("Internet Service (DSL/Fiber optic/No): "),
        input("Online Security (Yes/No/No internet service): "),
        input("Online Backup (Yes/No/No internet service): "),
        input("Device Protection (Yes/No/No internet service): "),
        input("Tech Support (Yes/No/No internet service): "),
        input("Streaming TV (Yes/No/No internet service): "),
        input("Streaming Movies (Yes/No/No internet service): "),
        input("Contract (Month-to-month/One year/Two year): "),
        input("Paperless Billing (Yes/No): "),
        input("Payment Method (Electronic check/Mailed check/Bank transfer (automatic)/Credit card (automatic)): "),
        float(input("Monthly Charges (in K): ")),
        float(input("Total Charges (in K): "))
    ]

    columns = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges"
    ]

    # Create dataframe
    new_data = pd.DataFrame([inputs], columns=columns)

    # --------------------------------
    # CatBoost data: original 19 columns
    # --------------------------------
    cat_data = new_data.copy()

    # --------------------------------
    # Data for LR / RF / XGB / ANN
    # --------------------------------

    # One-hot encoding
    new_data = pd.get_dummies(new_data)

    # Match exact training columns
    new_data = new_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Scale numerical columns
    scale_cols = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    new_data_scaled = new_data.copy()

    new_data_scaled[scale_cols] = scaler.transform(
        new_data[scale_cols]
    )

    # --------------------------------
    # Predictions
    # --------------------------------

    lr_pred = lr.predict(new_data_scaled)[0]

    rf_pred = rf.predict(new_data_scaled)[0]

    xgb_pred = xgb.predict(new_data_scaled)[0]

    cb_pred = cb.predict(cat_data)[0]

    # ANN
    ann_prob = model.predict(
        new_data_scaled,
        verbose=0
    )[0][0]

    ann_pred = int(ann_prob >= 0.5)

    # --------------------------------
    # Display results
    # --------------------------------

    print("\n========== CHURN PREDICTIONS ==========")

    print(
        "Logistic Regression :",
        "Churn" if lr_pred == 1 else "No Churn"
    )

    print(
        "Random Forest       :",
        "Churn" if rf_pred == 1 else "No Churn"
    )

    print(
        "XGBoost             :",
        "Churn" if xgb_pred == 1 else "No Churn"
    )

    print(
        "CatBoost            :",
        "Churn" if cb_pred == 1 else "No Churn"
    )

    print(
        "ANN                 :",
        "Churn" if ann_pred == 1 else "No Churn"
    )

    print(
        "\nANN Churn Probability:",
        round(ann_prob * 100, 2),
        "%"
    )


predict_churn()
