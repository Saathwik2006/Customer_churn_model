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
