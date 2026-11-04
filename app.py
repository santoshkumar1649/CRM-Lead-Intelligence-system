
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ---------- Page Configuration ----------
st.set_page_config(
    page_title="CRM Lead Intelligence",
    page_icon="📊",
    layout="wide",
)

# ---------- Constants ----------
MODEL_PATH = Path(__file__).parent / "models" / "crm_lead_model.pkl"
PRIORITY_THRESHOLD = 0.20

JOBS = [
    "admin.",
    "blue-collar",
    "entrepreneur",
    "housemaid",
    "management",
    "retired",
    "self-employed",
    "services",
    "student",
    "technician",
    "unemployed",
    "unknown",
]

MARITAL_STATUSES = ["divorced", "married", "single", "unknown"]

EDUCATION_LEVELS = [
    "basic.4y",
    "basic.6y",
    "basic.9y",
    "high.school",
    "illiterate",
    "professional.course",
    "university.degree",
    "unknown",
]

YES_NO_UNKNOWN = ["no", "yes", "unknown"]
CONTACT_TYPES = ["cellular", "telephone"]
DAYS = ["mon", "tue", "wed", "thu", "fri"]
MONTHS = [
    "mar", "apr", "may", "jun", "jul",
    "aug", "sep", "oct", "nov", "dec",
]
OUTCOMES = ["failure", "nonexistent", "success"]


# ---------- Load Model ----------
@st.cache_resource
def load_model(model_path: str):
    """Load and cache the trained model."""
    return joblib.load(model_path)


try:
    if not MODEL_PATH.is_file():
        st.error(
            f"Model file not found: `{MODEL_PATH}`. "
            "Please ensure the model exists in the models folder."
        )
        st.stop()

    model = load_model(str(MODEL_PATH))

    if not hasattr(model, "predict_proba"):
        st.error(
            "The loaded model does not support probability predictions."
        )
        st.stop()

except FileNotFoundError:
    st.error("The model file could not be found.")
    st.stop()

except Exception:
    st.error(
        "The model could not be loaded. Check that the model file is valid "
        "and all required Python dependencies are installed."
    )
    st.stop()


# ---------- Styling ----------
st.markdown(
    """
    <style>
    .block-container {
        width: 100%;
        max-width: 1400px;
        padding: 2rem 3rem;
    }

    .stNumberInput,
    .stSelectbox,
    .stTextInput {
        width: 100%;
    }

    div[data-testid="stFormSubmitButton"] > button {
        background-color: #2563EB;
        color: white;
        border: 1px solid #2563EB;
        transition: background-color 0.2s ease;
    }

    div[data-testid="stFormSubmitButton"] > button:hover {
        background-color: #1D4ED8;
        border-color: #1D4ED8;
        color: white;
    }

    div[data-testid="stFormSubmitButton"] > button:focus {
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.3);
    }

    @media (max-width: 768px) {
        .block-container {
            padding: 1rem;
        }

        h1 {
            font-size: 1.8rem !important;
        }

        h2 {
            font-size: 1.4rem !important;
        }

        [data-testid="stMetricValue"] {
            font-size: 1.8rem;
        }
    }

    @media (max-width: 480px) {
        .block-container {
            padding: 0.5rem;
        }

        h1 {
            font-size: 1.5rem !important;
        }

        div[data-testid="stFormSubmitButton"] > button {
            width: 100%;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Header ----------
st.title("📊 CRM Lead Intelligence")
st.caption("Predict conversion probability and prioritize sales leads.")
st.divider()


# ---------- Lead Input Form ----------
with st.form("lead_form"):
    st.subheader("Lead Profile")
    st.caption("Enter the lead's demographic and account information.")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=35,
            step=1,
        )

        job = st.selectbox("Job", JOBS)

        marital = st.selectbox(
            "Marital Status",
            MARITAL_STATUSES,
        )

    with col2:
        education = st.selectbox(
            "Education",
            EDUCATION_LEVELS,
        )

        default = st.selectbox(
            "Credit in Default",
            YES_NO_UNKNOWN,
        )

        housing = st.selectbox(
            "Housing Loan",
            YES_NO_UNKNOWN,
        )

    with col3:
        loan = st.selectbox(
            "Personal Loan",
            YES_NO_UNKNOWN,
        )

        contact = st.selectbox(
            "Contact Type",
            CONTACT_TYPES,
        )

        day_of_week = st.selectbox(
            "Day of Week",
            DAYS,
        )

    st.divider()

    st.subheader("Campaign Details")
    st.caption("Enter information about the current and previous campaigns.")

    col4, col5, col6 = st.columns(3)

    with col4:
        month = st.selectbox("Month", MONTHS)

        campaign = st.number_input(
            "Contacts During This Campaign",
            min_value=1,
            max_value=1000,
            value=1,
            step=1,
        )

    with col5:
        pdays = st.number_input(
            "Days Since Previous Contact (999 = never contacted)",
            min_value=0,
            max_value=999,
            value=999,
            step=1,
        )

        previous = st.number_input(
            "Previous Campaign Contacts",
            min_value=0,
            max_value=1000,
            value=0,
            step=1,
        )

    with col6:
        poutcome = st.selectbox(
            "Previous Campaign Outcome",
            OUTCOMES,
        )

    st.divider()

    st.subheader("Economic Indicators")
    st.caption("Provide the economic values associated with the lead's campaign.")

    col7, col8, col9 = st.columns(3)

    with col7:
        emp_var_rate = st.number_input(
            "Employment Variation Rate",
            value=0.0,
            format="%.3f",
        )

    with col8:
        cons_price_idx = st.number_input(
            "Consumer Price Index",
            value=93.0,
            format="%.3f",
        )

    with col9:
        cons_conf_idx = st.number_input(
            "Consumer Confidence Index",
            value=-40.0,
            format="%.3f",
        )

    col10, col11 = st.columns(2)

    with col10:
        euribor3m = st.number_input(
            "Euribor 3-Month Rate",
            value=4.0,
            format="%.3f",
        )

    with col11:
        employees = st.number_input(
            "Number of Employees",
            min_value=0,
            max_value=1_000_000,
            value=5000,
            step=100,
        )

    submitted = st.form_submit_button(
        "Score Lead",
        type="primary",
        use_container_width=True,
    )


# ---------- Prediction ----------
if submitted:
    try:
        lead = pd.DataFrame(
            [
                {
                    "age": int(age),
                    "job": job,
                    "marital": marital,
                    "education": education,
                    "default": default,
                    "housing": housing,
                    "loan": loan,
                    "contact": contact,
                    "month": month,
                    "day_of_week": day_of_week,
                    "campaign": int(campaign),
                    "pdays": int(pdays),
                    "previous": int(previous),
                    "poutcome": poutcome,
                    "emp.var.rate": float(emp_var_rate),
                    "cons.price.idx": float(cons_price_idx),
                    "cons.conf.idx": float(cons_conf_idx),
                    "euribor3m": float(euribor3m),
                    "nr.employed": float(employees),
                }
            ]
        )

        # Confirm that the model returns a valid probability.
        probabilities = model.predict_proba(lead)

        if probabilities.shape != (1, 2):
            raise ValueError(
                "Expected a binary classifier with two probability columns."
            )

        probability = float(probabilities[0, 1])

        if not 0.0 <= probability <= 1.0:
            raise ValueError("The model returned an invalid probability.")

        priority = (
            "High"
            if probability >= PRIORITY_THRESHOLD
            else "Low"
        )

        # ---------- Results ----------
        st.divider()
        st.subheader("Lead Score")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric(
                label="Conversion Probability",
                value=f"{probability:.2%}",
            )

        with result_col2:
            if priority == "High":
                st.markdown(
                    """
                    <div style="
                        background-color: #166534;
                        color: white;
                        padding: 12px 16px;
                        border-radius: 10px;
                        font-weight: 600;
                        margin-top: 8px;
                    ">
                        🟢 HIGH PRIORITY
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div style="
                        background-color: #374151;
                        color: white;
                        padding: 12px 16px;
                        border-radius: 10px;
                        font-weight: 600;
                        margin-top: 8px;
                    ">
                        ⚪ LOW PRIORITY
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.progress(probability)

        if priority == "High":
            st.success(
                "High-priority lead: consider timely sales follow-up."
            )
        else:
            st.info(
                "Low-priority lead: consider follow-up based on "
                "available sales capacity."
            )

        st.caption(
            "This score is a model estimate based on the Bank Marketing "
            "dataset, not a guarantee of conversion."
        )

    except (ValueError, TypeError, IndexError, AttributeError) as error:
        st.error(
            "Unable to score this lead. Please check the input values "
            "and verify that the model matches the expected data schema."
        )
        st.caption(f"Technical details: {error}")

    except Exception:
        st.error(
            "An unexpected prediction error occurred. "
            "Please check your model and try again."
        )
