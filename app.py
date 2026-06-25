import streamlit as st
import requests

st.set_page_config(page_title="Churn Predictor", page_icon="📊")

st.title("📊 Customer Churn Predictor")
st.markdown("Fill in customer details to predict if they will churn.")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone = st.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_sec = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

with col2:
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly_charges = st.slider("Monthly Charges ($)", 0.0, 120.0, 65.0)
    total_charges = st.slider("Total Charges ($)", 0.0, 8000.0, float(tenure * monthly_charges))

def encode(val, options):
    return options.index(val)

if st.button("🔮 Predict Churn", use_container_width=True):
    data = {
        "gender": encode(gender, ["Female", "Male"]),
        "SeniorCitizen": encode(senior, ["No", "Yes"]),
        "Partner": encode(partner, ["No", "Yes"]),
        "Dependents": encode(dependents, ["No", "Yes"]),
        "tenure": tenure,
        "PhoneService": encode(phone, ["No", "Yes"]),
        "MultipleLines": encode(multiple_lines, ["No", "No phone service", "Yes"]),
        "InternetService": encode(internet, ["DSL", "Fiber optic", "No"]),
        "OnlineSecurity": encode(online_sec, ["No", "No internet service", "Yes"]),
        "OnlineBackup": encode(online_backup, ["No", "No internet service", "Yes"]),
        "DeviceProtection": encode(device_protection, ["No", "No internet service", "Yes"]),
        "TechSupport": encode(tech_support, ["No", "No internet service", "Yes"]),
        "StreamingTV": encode(streaming_tv, ["No", "No internet service", "Yes"]),
        "StreamingMovies": encode(streaming_movies, ["No", "No internet service", "Yes"]),
        "Contract": encode(contract, ["Month-to-month", "One year", "Two year"]),
        "PaperlessBilling": encode(paperless, ["No", "Yes"]),
        "PaymentMethod": encode(payment, ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"]),
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    response = requests.post(
        "https://churn-mlops-82cq.onrender.com/predict",
        json=data
    )
    result = response.json()

    prob = result["churn_probability"]
    churn = result["churn"]

    st.divider()
    if churn:
        st.error(f"⚠️ This customer is likely to churn! Probability: {prob*100:.1f}%")
    else:
        st.success(f"✅ This customer is NOT likely to churn. Probability: {prob*100:.1f}%")

    st.progress(prob)
