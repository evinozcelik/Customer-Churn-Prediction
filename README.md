# Customer Churn Prediction

This project focuses on predicting customer churn for a telecommunications company using machine learning techniques. The goal is to identify customers who are likely to leave the service so that businesses can take proactive actions to retain them.

Customer churn prediction is a common problem in subscription-based industries where losing customers directly affects revenue. Machine learning models can help companies detect churn risk early and design effective retention strategies.

Dataset: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

🌐 [Try the Application](https://customer-churn-prediction-tool.streamlit.app/#contract-and-billing)

---

# Project Overview

The project follows a typical machine learning pipeline:

1. Exploratory Data Analysis (EDA)
2. Baseline Model Training
3. Feature Engineering
4. Hyperparameter Tuning
5. Model Evaluation
6. Experiment Tracking with MLflow
7. Model Saving and Reproducibility

The dataset was analyzed and transformed to extract meaningful patterns that help predict customer churn.

---

# Dataset

The dataset contains **7043 customer records** with multiple features describing customer demographics, services, and billing information.

Main feature categories include:

**Customer Information**
- gender
- SeniorCitizen
- Partner
- Dependents

**Service Usage**
- PhoneService
- InternetService
- OnlineSecurity
- OnlineBackup
- StreamingTV
- StreamingMovies

**Account Information**
- Contract
- PaymentMethod
- PaperlessBilling

**Billing Features**
- tenure
- MonthlyCharges
- TotalCharges

The target variable is:

**Churn**
- Yes → customer left
- No → customer stayed

---

# Exploratory Data Analysis (EDA)

EDA was performed to understand the structure of the dataset and detect important patterns.

Key observations:

- The dataset contains **7043 observations and 21 original features**
- The churn rate is **around 26%**, indicating class imbalance
- Most features are **categorical service-related variables**
- Customers with **shorter tenure tend to churn more frequently**
- Certain **contract types and payment methods show higher churn tendencies**

EDA helped identify relationships between customer behavior and churn.

---

# Feature Engineering

Feature engineering was applied to improve the predictive capability of the models.

New features were created based on customer behavior and billing information.

Examples include:

- **tenure_group** – grouping customers based on subscription duration
- **charges_ratio** – average spending behavior derived from billing features
- service-based feature transformations

After feature engineering:

- Original feature count: **21**
- Final feature count: **43**

These additional features help models capture more meaningful patterns related to churn behavior.

---

# Machine Learning Models

Two models were trained and evaluated:

### Logistic Regression
A simple and interpretable baseline model used as a benchmark.

### LightGBM
A gradient boosting model capable of capturing complex relationships between features.

Hyperparameter tuning was performed using **GridSearchCV** to optimize LightGBM performance.

---

# Model Results

| Model | Train AUC | Test AUC | Precision | Recall | F1 Score |
|------|------|------|------|------|------|
| Logistic Regression | 0.8449 | 0.8397 | 0.5103 | 0.7914 | 0.6205 |
| LightGBM | 0.9880 | 0.8305 | 0.5544 | 0.7219 | 0.6271 |

Logistic Regression demonstrated strong recall, meaning it successfully identifies many churn customers.

LightGBM provided better precision and overall F1-score, indicating a stronger balance between correctly detecting churners and avoiding unnecessary predictions.

---

# Final Model

Based on the evaluation results, **LightGBM was selected as the final model** due to its balanced predictive performance and ability to capture complex feature relationships.

The final model was saved using **joblib** for future inference and deployment.

---

# Experiment Tracking

All experiments were tracked using **MLflow**, including:

- model parameters
- evaluation metrics
- trained models

This ensures experiment reproducibility and easy comparison between different models.

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- LightGBM
- MLflow
- Matplotlib
- Seaborn
- Joblib

---

# Business Impact

Customer churn prediction allows companies to:

- Identify customers at risk of leaving
- Apply targeted retention strategies
- Reduce revenue loss
- Improve customer relationship management

Machine learning models provide data-driven insights that support better business decisions.

---

# Future Improvements

Possible improvements for the project include:

- Testing additional models such as XGBoost or CatBoost
- Performing more advanced feature engineering
- Applying cross-validation strategies for more robust evaluation
- Building a deployment pipeline or API for real-time predictions
