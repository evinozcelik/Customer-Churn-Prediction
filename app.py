import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")
st.markdown(
    """
    <style>
    .stApp {
        background-color: #F5F7FA;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource
def load_resources():
    model_path = "models/lightgbm_fe_model.pkl" 
    features_path = "models/feature_columns.pkl"  
    train_data_path = "data/X_train_fe.csv" 
    
    model, feature_columns, cat_mappings = None, None, {}
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    if os.path.exists(features_path):
        feature_columns = joblib.load(features_path)
        
    if os.path.exists(train_data_path):
        X_train = pd.read_csv(train_data_path)
        categorical_features = X_train.select_dtypes(include=["object"]).columns
        for col in categorical_features:
            unique_vals = X_train[col].dropna().unique()
            mapping = {val: ide for ide, val in enumerate(unique_vals)}
            cat_mappings[col] = mapping
            
    return model, feature_columns, cat_mappings

model, feature_columns, cat_mappings = load_resources()


st.title("📊 Telecom Customer Churn Prediction Tool")
st.markdown("Fill out the customer information form below to calculate the probability of the customer churning (leaving the company).")
st.write("---")

if model is None:
    st.error("Model file not found! Please check the model_path.")
else:
    
    with st.form("customer_form"):
        
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("👤 Personal Information")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen?", [0, 1])
            partner = st.selectbox("Has Partner?", ["Yes", "No"])
            dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
            tenure = st.slider("Tenure (Months)", 0, 75, 12)
            
        with col2:
            st.subheader("⚙️ Service Details")
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
            
        with col3:
            st.subheader("💳 Contract & Billing")
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=500.0)

        st.write("---")
        
        submitted = st.form_submit_button("Calculate Churn Probability", use_container_width=True)

    
    if submitted:
        user_data = {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }
        
        input_df = pd.DataFrame(user_data, index=[0])

        
        def apply_feature_engineering(df):
            df['Average_Monthly_Charge'] = np.where(df['tenure'] == 0, 
                                                    df['MonthlyCharges'], 
                                                    df['TotalCharges'] / df['tenure'])
            df['Charge_Difference'] = df['MonthlyCharges'] - df['Average_Monthly_Charge']

            df['Is_Streaming_User'] = ((df['StreamingTV'] == 'Yes') | (df['StreamingMovies'] == 'Yes')).astype(int)
            df['Security_and_Support_Risk'] = ((df['OnlineSecurity'] == 'No') & (df['TechSupport'] == 'No')).astype(int)
            df['Is_Family'] = ((df['Partner'] == 'Yes') | (df['Dependents'] == 'Yes')).astype(int)
            df['Is_AutoPay'] = df['PaymentMethod'].str.contains('automatic').astype(int)

            df['Risk_Segment'] = ((df['Contract'] == 'Month-to-month') & (df['PaymentMethod'] == 'Electronic check')).astype(int)
            df['short_contract_low_tenure'] = ((df['Contract'] == 'Month-to-month') & (df['tenure'] < 12)).astype(int)

            contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
            df["contract_length"] = df["Contract"].map(contract_map)

            df = df.drop('Contract', axis=1)

            bins = [-1, 12, 36, 60, np.inf]
            labels = ['0-1 Year', '1-3 Years', '3-5 Years', '5+ Years']
            df['Tenure_Cohorts'] = pd.cut(df['tenure'], bins=bins, labels=labels).astype(str)

            service_cols = [
                "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup", 
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
            ]
            df["total_services"] = (df[service_cols] == "Yes").sum(axis=1)
            df["charge_per_service"] = df["MonthlyCharges"] / (df["total_services"] + 1)
            df["has_internet"] = (df["InternetService"] != "No").astype(int)

            medianMonthlyCharges=70.5
            df['senior_high_charge'] = ((df['SeniorCitizen'] == 1) & (df['MonthlyCharges'] > medianMonthlyCharges)).astype(int)
            
            return df

        processed_df = apply_feature_engineering(input_df)

        if cat_mappings:
            for col, mapping in cat_mappings.items():
                if col in processed_df.columns:
                    processed_df[col] = processed_df[col].map(mapping)
        else:
            st.error("⚠️ data/X_train_fe.csv file not found! Categorical mappings will be missing.")

        for col in feature_columns:
            if col not in processed_df.columns:
                processed_df[col] = 0

        processed_df = processed_df[feature_columns]

        
        with st.spinner("AI is analyzing customer data..."):
            
            input_array = processed_df.astype(float).values
            
            prediction = model.predict(input_array)
            prediction_proba = model.predict_proba(input_array)[0][1]

            st.write("---")
            st.subheader("Prediction Result")
            
            if prediction[0] == 1:
                st.error(f"⚠️ **HIGH RISK: The customer is very likely to churn!**")
            else:
                st.success(f"✅ **SAFE: The customer is likely to stay.**")
                
            st.info(f"Churn Probability: **{prediction_proba * 100:.2f}%**")
            
            

            st.progress(prediction_proba)




