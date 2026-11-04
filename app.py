
import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/crm_lead_model.pkl")

st.set_page_config(
    page_title="CRM Lead Intelligence",
    page_icon="📊",
    layout="wide"
)

st.title("CRM Lead Intelligence System")
st.write("Predict lead conversion probability and prioritize sales follow-ups.")

st.header("Enter Lead Details")

with st.form("lead_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        job = st.selectbox("Job", [
            "admin.", "blue-collar", "entrepreneur", "housemaid",
            "management", "retired", "self-employed", "services",
            "student", "technician", "unemployed", "unknown"
        ])
        marital = st.selectbox("Marital Status", [
            "divorced", "married", "single", "unknown"
        ])
        education = st.selectbox("Education", [
            "basic.4y", "basic.6y", "basic.9y", "high.school",
            "illiterate", "professional.course", "university.degree", "unknown"
        ])

    with col2:
        default = st.selectbox("Credit in Default", ["no", "yes", "unknown"])
        housing = st.selectbox("Housing Loan", ["no", "yes", "unknown"])
        loan = st.selectbox("Personal Loan", ["no", "yes", "unknown"])
        contact = st.selectbox("Contact Type", ["cellular", "telephone"])
        month = st.selectbox("Month", [
            "mar", "apr", "may", "jun", "jul", "aug",
            "sep", "oct", "nov", "dec"
        ])

    with col3:
        day_of_week = st.selectbox(
            "Day of Week", ["mon", "tue", "wed", "thu", "fri"]
        )
        campaign = st.number_input(
            "Contacts During This Campaign", min_value=1, value=1
        )
        pdays = st.number_input(
            "Days Since Previous Contact (999 = never contacted)",
            min_value=0, value=999
        )
        previous = st.number_input(
            "Previous Campaign Contacts", min_value=0, value=0
        )
        poutcome = st.selectbox(
            "Previous Campaign Outcome",
            ["failure", "nonexistent", "success"]
        )

    st.subheader("Economic Indicators")
    col4, col5, col6 = st.columns(3)

    with col4:
        emp_var_rate = st.number_input("Employment Variation Rate", value=0.0)
    with col5:
        cons_price_idx = st.number_input("Consumer Price Index", value=93.0)
    with col6:
        cons_conf_idx = st.number_input("Consumer Confidence Index", value=-40.0)

    col7, col8 = st.columns(2)
    with col7:
        euribor3m = st.number_input("Euribor 3-Month Rate", value=4.0)
    with col8:
        nr_employed = st.number_input("Number of Employees", value=5000.0)

    submitted = st.form_submit_button("Score Lead")

if submitted:
    lead = pd.DataFrame([{
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "month": month,
        "day_of_week": day_of_week,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome,
        "emp.var.rate": emp_var_rate,
        "cons.price.idx": cons_price_idx,
        "cons.conf.idx": cons_conf_idx,
        "euribor3m": euribor3m,
        "nr.employed": nr_employed
    }])

    probability = model.predict_proba(lead)[0, 1]
    threshold = 0.20
    priority = "High" if probability >= threshold else "Low"

    st.subheader("Lead Score")
    st.metric("Conversion Probability", f"{probability:.2%}")
    st.metric("Priority", priority)
    st.progress(float(probability))
