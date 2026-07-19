import os

import pandas as pd
import streamlit as st

from model_utils import fit_pipeline, load_data, predict_single

st.set_page_config(
    page_title="CreditOptima - Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "loan_approval_data.csv")


@st.cache_resource(show_spinner="Training models on the loan dataset...")
def get_artifacts():
    df = load_data(DATA_PATH)
    return df, fit_pipeline(df)


df, artifacts = get_artifacts()

st.title("🏦 CreditOptima - Loan Approval Predictor")
st.caption(
    "Fill in an applicant's details to predict whether their loan would be approved. "
    "Models are trained on startup from `loan_approval_data.csv`."
)

tab_predict, tab_models = st.tabs(["🔮 Predict", "📊 Model Comparison"])

# PREDICT TAB

with tab_predict:
    model_name = st.selectbox(
        "Model to use for prediction",
        options=list(artifacts["models"].keys()),
        index=0,
    )

    with st.form("prediction_form"):
        st.subheader("Applicant details")
        c1, c2, c3 = st.columns(3)

        with c1:
            applicant_income = st.number_input("Applicant Income", min_value=0, value=10000, step=500)
            coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=3000, step=500)
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            dependents = st.number_input("Dependents", min_value=0, max_value=10, value=1)
            credit_score = st.slider("Credit Score", min_value=300, max_value=850, value=680)
            existing_loans = st.number_input("Existing Loans", min_value=0, max_value=10, value=1)

        with c2:
            dti_ratio = st.slider("DTI Ratio (debt-to-income)", min_value=0.0, max_value=1.0, value=0.35, step=0.01)
            savings = st.number_input("Savings", min_value=0, value=10000, step=500)
            collateral_value = st.number_input("Collateral Value", min_value=0, value=25000, step=500)
            loan_amount = st.number_input("Loan Amount", min_value=0, value=20000, step=500)
            loan_term = st.number_input("Loan Term (months)", min_value=1, max_value=480, value=60)

        with c3:
            employment_status = st.selectbox(
                "Employment Status", ["Salaried", "Self-employed", "Contract", "Unemployed"]
            )
            marital_status = st.selectbox("Marital Status", ["Married", "Single"])
            loan_purpose = st.selectbox(
                "Loan Purpose", ["Personal", "Car", "Business", "Home", "Education"]
            )
            property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
            education_level = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
            gender = st.selectbox("Gender", ["Male", "Female"])
            employer_category = st.selectbox(
                "Employer Category", ["Private", "Government", "MNC", "Business", "Unemployed"]
            )

        submitted = st.form_submit_button("Predict Loan Approval", use_container_width=True)

    if submitted:
        raw_input = {
            "Applicant_Income": applicant_income,
            "Coapplicant_Income": coapplicant_income,
            "Employment_Status": employment_status,
            "Age": age,
            "Marital_Status": marital_status,
            "Dependents": dependents,
            "Credit_Score": credit_score,
            "Existing_Loans": existing_loans,
            "DTI_Ratio": dti_ratio,
            "Savings": savings,
            "Collateral_Value": collateral_value,
            "Loan_Amount": loan_amount,
            "Loan_Term": loan_term,
            "Loan_Purpose": loan_purpose,
            "Property_Area": property_area,
            "Education_Level": education_level,
            "Gender": gender,
            "Employer_Category": employer_category,
        }

        label, proba = predict_single(raw_input, artifacts, model_name)

        st.divider()
        if label == "Approved":
            st.success(f"✅ Prediction: **{label}**")
        else:
            st.error(f"❌ Prediction: **{label}**")

        st.metric("Estimated probability of approval", f"{proba * 100:.1f}%")
        st.progress(min(max(proba, 0.0), 1.0))
        st.caption(f"Predicted using **{model_name}**.")

# MODEL COMPARISON TAB

with tab_models:
    st.subheader("How each model performed on the held-out test set")

    rows = []
    for name, m in artifacts["metrics"].items():
        rows.append(
            {
                "Model": name,
                "Accuracy": m["Accuracy"],
                "Precision": m["Precision"],
                "Recall": m["Recall"],
                "F1 Score": m["F1"],
            }
        )
    metrics_df = pd.DataFrame(rows).set_index("Model")
    st.dataframe(metrics_df.style.format("{:.3f}").highlight_max(axis=0, color="#c6f6d5"), use_container_width=True)

    st.bar_chart(metrics_df)

    st.subheader("Confusion matrices")
    cols = st.columns(len(artifacts["metrics"]))
    for col, (name, m) in zip(cols, artifacts["metrics"].items()):
        with col:
            st.write(f"**{name}**")
            cm = pd.DataFrame(
                m["Confusion Matrix"],
                index=["Actual: No", "Actual: Yes"],
                columns=["Pred: No", "Pred: Yes"],
            )
            st.dataframe(cm, use_container_width=True)

