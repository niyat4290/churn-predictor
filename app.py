"""
app.py  —  Telco Customer Churn Predictor
Run locally:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📡",
    layout="wide",
)

# ── Load model artifacts ───────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model     = joblib.load("models/model.pkl")
    encoders  = joblib.load("models/encoders.pkl")
    explainer = joblib.load("models/explainer.pkl")
    features  = joblib.load("models/feature_names.pkl")
    return model, encoders, explainer, features

model, encoders, explainer, feature_names = load_artifacts()

# ── Helper ─────────────────────────────────────────────────────────────────────
def encode_input(raw: dict, encoders: dict) -> pd.DataFrame:
    df = pd.DataFrame([raw])
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col])
    return df[feature_names]

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("📡 Telco Customer Churn Predictor")
st.markdown(
    "Enter customer details on the left → get a **churn probability** and "
    "an **AI explanation** of what's driving the risk."
)
st.divider()

# ── Sidebar: user inputs ────────────────────────────────────────────────────────
st.sidebar.header("🧑 Customer Profile")

gender          = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior          = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner         = st.sidebar.selectbox("Has Partner?", ["Yes", "No"])
dependents      = st.sidebar.selectbox("Has Dependents?", ["No", "Yes"])
tenure          = st.sidebar.slider("Tenure (months)", 0, 72, 12)
phone_service   = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines  = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet        = st.sidebar.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
online_sec      = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
online_backup   = st.sidebar.selectbox("Online Backup",   ["No", "Yes", "No internet service"])
device_prot     = st.sidebar.selectbox("Device Protection",["No","Yes","No internet service"])
tech_support    = st.sidebar.selectbox("Tech Support",    ["No", "Yes", "No internet service"])
streaming_tv    = st.sidebar.selectbox("Streaming TV",    ["No", "Yes", "No internet service"])
streaming_mov   = st.sidebar.selectbox("Streaming Movies",["No", "Yes", "No internet service"])
contract        = st.sidebar.selectbox("Contract Type", ["Month-to-month","One year","Two year"])
paperless       = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment         = st.sidebar.selectbox("Payment Method", [
    "Electronic check","Mailed check",
    "Bank transfer (automatic)","Credit card (automatic)"
])
monthly_charges = st.sidebar.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
total_charges   = st.sidebar.slider("Total Charges ($)", 0.0, 8700.0, monthly_charges * tenure, step=10.0)

# ── Build raw input dict ────────────────────────────────────────────────────────
raw = {
    "gender": gender,
    "SeniorCitizen": 1 if senior == "Yes" else 0,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone_service,
    "MultipleLines": multiple_lines,
    "InternetService": internet,
    "OnlineSecurity": online_sec,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_prot,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_mov,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
}

# ── Predict ────────────────────────────────────────────────────────────────────
X_input  = encode_input(raw, encoders)
prob     = model.predict_proba(X_input)[0][1]
pred     = "🚨 Likely to Churn" if prob >= 0.5 else "✅ Likely to Stay"
color    = "#e74c3c" if prob >= 0.5 else "#27ae60"

# ── Results layout ─────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Prediction")
    st.markdown(
        f"<h2 style='color:{color}'>{pred}</h2>",
        unsafe_allow_html=True,
    )
    st.metric("Churn Probability", f"{prob*100:.1f}%")

    # Risk gauge bar
    fig_gauge, ax = plt.subplots(figsize=(4, 0.6))
    ax.barh(0, prob, color=color, height=0.4)
    ax.barh(0, 1 - prob, left=prob, color="#ecf0f1", height=0.4)
    ax.set_xlim(0, 1); ax.axis("off")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1)
    st.pyplot(fig_gauge, use_container_width=True)
    plt.close(fig_gauge)

with col2:
    st.subheader("🔍 Why this prediction? (SHAP Explanation)")
    shap_values = explainer.shap_values(X_input)

    # Waterfall-style bar chart
    vals  = shap_values[0]
    feats = feature_names
    df_shap = pd.DataFrame({"feature": feats, "shap": vals})
    df_shap = df_shap.reindex(df_shap["shap"].abs().sort_values(ascending=False).index).head(10)

    fig_shap, ax2 = plt.subplots(figsize=(7, 4))
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in df_shap["shap"]]
    ax2.barh(df_shap["feature"][::-1], df_shap["shap"][::-1], color=colors[::-1])
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("SHAP value  (red = pushes toward churn, blue = pushes toward stay)")
    ax2.set_title("Top 10 Feature Contributions")
    fig_shap.tight_layout()
    st.pyplot(fig_shap, use_container_width=True)
    plt.close(fig_shap)

st.divider()

# ── Global Feature Importance ──────────────────────────────────────────────────
with st.expander("📊 Global Feature Importance (across all customers)"):
    importances = model.feature_importances_
    df_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    df_imp = df_imp.sort_values("Importance", ascending=False).head(12)

    fig_imp, ax3 = plt.subplots(figsize=(7, 4))
    ax3.barh(df_imp["Feature"][::-1], df_imp["Importance"][::-1], color="#2ecc71")
    ax3.set_xlabel("Importance Score")
    ax3.set_title("Top 12 Most Important Features (XGBoost)")
    fig_imp.tight_layout()
    st.pyplot(fig_imp, use_container_width=True)
    plt.close(fig_imp)

# ── Dataset preview ────────────────────────────────────────────────────────────
with st.expander("🗃️ Raw Dataset Preview"):
    df_raw = pd.read_csv("Telco-Customer-Churn.csv")
    st.dataframe(df_raw.head(50), use_container_width=True)

st.caption("Built with XGBoost + SHAP · Telco Customer Churn dataset")