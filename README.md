# Telco Customer Churn Prediction

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

### Feature Description

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

### Target Distribution

| Churn Status | Count | Percentage |
| ------------ | ----: | ---------: |
| No           | 5,174 |     73.46% |
| Yes          | 1,869 |     26.54% |

The dataset is moderately imbalanced, with approximately **73% non-churn** and **27% churn** customers. This imbalance was considered during model evaluation by comparing multiple performance metrics, including Precision, Recall, and F1-score, rather than relying solely on Accuracy.


## Data Preprocessing

The Telco Customer Churn dataset was carefully preprocessed to ensure high-quality inputs for machine learning models.

### Dataset Overview

* **Rows:** 7,043
* **Target Variable:** `Churn`
* **Features:** Demographic, account, and service-related customer attributes.

### Preprocessing Steps

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


## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | CV Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| **Logistic Regression** | 80.55% | 65.72% | **55.88%** | **60.40%** | — | **80.46%** |
| **Random Forest** | 80.48% | **69.41%** | 47.33% | 56.28% | 83.99% | 79.97% |
| **XGBoost** | **80.70%** | 67.47% | 52.67% | 59.16% | **84.68%** | 80.42% |
| **CatBoost** | 80.62% | 67.24% | 52.67% | 59.07% | — | 79.79% |
| **ANN (DNN)** | 79.84% | 63.16% | 57.75% | 60.34% | 83.93% | — |

> **Note:** ROC-AUC was not calculated for Logistic Regression and CatBoost in the notebook, and CV accuracy was not calculated for the ANN. R² was excluded because it is not an appropriate metric for classification.

### Confusion Matrices

| Model | True Negative | False Positive | False Negative | True Positive |
|---|---:|---:|---:|---:|
| **Logistic Regression** | 926 | 109 | 165 | 209 |
| **Random Forest** | 957 | 78 | 197 | 177 |
| **XGBoost** | 940 | 95 | 177 | 197 |
| **CatBoost** | 939 | 96 | 177 | 197 |
| **ANN (DNN)** | 909 | 126 | **158** | **216** |

## Interesting Observations

- **XGBoost achieved the highest accuracy (80.70%) and ROC-AUC (84.68%)**, making it the strongest overall model in this experiment.
- **Random Forest achieved the highest precision (69.41%)**. However, its recall was only 47.33%, meaning it missed a relatively large number of actual churners.
- **ANN achieved the highest recall (57.75%)**, correctly identifying **216 of 374 actual churners**. This can be valuable when the business priority is to identify as many potential churners as possible.
- **Logistic Regression performed surprisingly well**, achieving 80.55% accuracy and the highest F1 score (60.40%) among the models tested.
- The boosting models did **not dramatically outperform** the simpler models on this tabular dataset. All models achieved approximately **80% accuracy**.
- Although the ANN had slightly lower accuracy than XGBoost, it achieved **higher recall (57.75% vs. 52.67%)**, making it potentially more suitable when missing a potential churner is more costly than generating some false positives.
- The cross-validation accuracy of Logistic Regression, Random Forest, and XGBoost was close to their respective test accuracy, suggesting relatively consistent performance across the evaluation splits.
