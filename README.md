### Telco Customer Churn Prediction

This project uses the **Telco Customer Churn** dataset to predict whether a customer is likely to discontinue their telecommunications service. Customer churn prediction enables businesses to identify at-risk customers and implement targeted retention strategies.

## Dataset Information

| Attribute       | Value                        |
| --------------- | ---------------------------- |
| Dataset         | Telco Customer Churn         |
| Source          | IBM Sample Data              |
| Total Records   | 7,043                        |
| Features        | 20 Input Features + 1 Target |
| Target Variable | `Churn`                      |
| Task            | Binary Classification        |

## Feature Description

| Feature          | Description                                               |
| ---------------- | --------------------------------------------------------- |
| customerID       | Unique customer identifier                                |
| gender           | Customer gender                                           |
| SeniorCitizen    | Whether the customer is a senior citizen                  |
| Partner          | Whether the customer has a partner                        |
| Dependents       | Whether the customer has dependents                       |
| tenure           | Number of months the customer has stayed with the company |
| PhoneService     | Phone service subscription                                |
| MultipleLines    | Multiple phone lines subscription                         |
| InternetService  | Type of internet service                                  |
| OnlineSecurity   | Online security service                                   |
| OnlineBackup     | Online backup service                                     |
| DeviceProtection | Device protection plan                                    |
| TechSupport      | Technical support subscription                            |
| StreamingTV      | Streaming TV subscription                                 |
| StreamingMovies  | Streaming Movies subscription                             |
| Contract         | Customer contract type                                    |
| PaperlessBilling | Paperless billing status                                  |
| PaymentMethod    | Payment method used                                       |
| MonthlyCharges   | Monthly billing amount                                    |
| TotalCharges     | Total amount charged to the customer                      |
| **Churn**        | Whether the customer left the company (Target Variable)   |

## Target Distribution

| Churn Status | Count | Percentage |
| ------------ | ----: | ---------: |
| No           | 5,174 |     73.46% |
| Yes          | 1,869 |     26.54% |

The dataset is moderately imbalanced, with approximately **73% non-churn** and **27% churn** customers. This imbalance was considered during model evaluation by comparing multiple performance metrics, including Precision, Recall, and F1-score, rather than relying solely on Accuracy.


# Data Preprocessing

The Telco Customer Churn dataset was carefully preprocessed to ensure high-quality inputs for machine learning models.

## Dataset Overview

* **Rows:** 7,043
* **Target Variable:** `Churn`
* **Features:** Demographic, account, and service-related customer attributes.

## Preprocessing Steps

* Removed the unique identifier column (`customerID`) as it does not contribute to prediction.
* Converted `TotalCharges` from object to numeric datatype.
* Filled missing values in `TotalCharges` using the median.
* Converted the target variable (`Churn`) into binary values (`0 = No`, `1 = Yes`).
* Applied **Ordinal Encoding** to the `Contract` feature.
* Applied **One-Hot Encoding** to all remaining nominal categorical features.
* Standardized numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) using `StandardScaler`.
* Created a separate unencoded dataset for CatBoost to preserve categorical features.
* Performed an **80-20 train-test split** using `stratify` to maintain the original class distribution.
* Used **5-Fold Cross Validation** to evaluate model generalization.


# Baseline Model Performance

The following baseline models were trained using default or near-default hyperparameters.

| Model               | Accuracy | Precision |     Recall |   F1 Score | R² Score | Mean CV Accuracy |
| ------------------- | -------: | --------: | ---------: | ---------: | -------: | ---------------: |
| Logistic Regression |   0.8055 |    0.6572 | **0.5588** | **0.6040** |   0.0026 |       **0.8046** |
| Random Forest       |   0.7807 |    0.6117 |     0.4759 |     0.5353 |  -0.1248 |           0.7883 |
| XGBoost             |   0.7779 |    0.5950 |     0.5107 |     0.5496 |  -0.1393 |           0.7913 |
| CatBoost            |   0.7999 |    0.6523 |     0.5267 |     0.5828 |  -0.0265 |           0.7860 |

### Observation

* Logistic Regression provided the strongest baseline performance.
* Tree-based ensemble methods required hyperparameter tuning to unlock their full potential.
* CatBoost performed competitively despite requiring no categorical encoding.


# Tuned Model Performance

| Model                 |   Accuracy |  Precision |     Recall |   F1 Score |   R² Score | Mean CV Accuracy |
| --------------------- | ---------: | ---------: | ---------: | ---------: | ---------: | ---------------: |
| Logistic Regression   |     0.8055 |     0.6572 | **0.5588** | **0.6040** |     0.0026 |       **0.8046** |
| Random Forest (Tuned) | **0.8091** | **0.6829** |     0.5241 |     0.5930 | **0.0208** |           0.8031 |
| XGBoost (Tuned)       |     0.8070 |     0.6747 |     0.5267 |     0.5916 |     0.0099 |           0.8042 |
| CatBoost (Tuned)      |     0.8062 |     0.6724 |     0.5267 |     0.5907 |     0.0063 |                — |

## Balanced Logistic Regression (Experiment)

| Model                                           | Accuracy | Precision |     Recall |   F1 Score |
| ----------------------------------------------- | -------: | --------: | ---------: | ---------: |
| Logistic Regression (`class_weight="balanced"`) |   0.7381 |    0.5043 | **0.7834** | **0.6136** |

This additional experiment demonstrates the trade-off between overall accuracy and the ability to correctly identify churning customers.


