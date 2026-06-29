import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="FraudGuard - Fake Payment Detector", page_icon="🛡️", layout="centered")

st.title("🛡️ FraudGuard: Fake Payment Detector")
st.write("Machine Learning se fake aur fraudulent transactions ko detect karein.")
st.markdown("---")

@st.cache_resource
def train_model():
    np.random.seed(42)
    n_samples = 2000
    amount = np.random.uniform(5, 5000, n_samples)
    distance = np.random.uniform(0, 500, n_samples)
    is_intl = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
    failed_attempts = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.7, 0.2, 0.07, 0.03])
    
    is_fraud = np.where(
        (amount > 4000) & (failed_attempts >= 2) |
        (is_intl == 1) & (distance > 300) |
        (amount > 4800), 1, 0
    )
    
    df = pd.DataFrame({
        'Amount': amount,
        'Distance': distance,
        'Is_International': is_intl,
        'Failed_Attempts': failed_attempts,
        'Class': is_fraud
    })
    
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    return model

model = train_model()

st.subheader("💳 Transaction Details Enter Karein")
col1, col2 = st.columns(2)

with col1:
    amount = st.number_input("Transaction Amount (₹):", min_value=1.0, max_value=50000.0, value=500.0, step=10.0)
    distance = st.number_input("Ghar se Distance (KM me):", min_value=0.0, max_value=2000.0, value=10.0, step=5.0)

with col2:
    intl_status = st.selectbox("Kya IP Address International hai?", ["No", "Yes"])
    failed_attempts = st.slider("Pichle 1 ghante me Failed PIN Attempts:", 0, 5, 0)

is_intl = 1 if intl_status == "Yes" else 0

st.markdown("---")
if st.button("🚨 Verify Payment / Check Fraud"):
    user_data = np.array([[amount, distance, is_intl, failed_attempts]])
    prediction = model.predict(user_data)
    probability = model.predict_proba(user_data)[0][1] * 100
    
    st.subheader("🔍 Analysis Result:")
    if prediction[0] == 1 or probability > 50:
        st.error("🔴 ALERT: Fake/Fraudulent Payment Detected!")
        st.metric(label="Fraud Risk Probability", value=f"{probability:.2f}%")
    else:
        st.success("🟢 SAFE: Yeh Payment Real hai.")
        st.metric(label="Fraud Risk Probability", value=f"{probability:.2f}%")
