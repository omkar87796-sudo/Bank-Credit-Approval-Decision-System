import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ==================================================
# Page Configuration
# ==================================================
st.set_page_config(
    page_title="Bank Credit Approval System",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Bank Credit Approval Decision System")
st.caption("AI-powered decision support for loan applications")

# ==================================================
# Load Trained Pipeline
# ==================================================
@st.cache_resource
def load_pipeline():
    return joblib.load("credit_risk_pipeline.pkl")   # ✅ FIXED NAME

pipeline = load_pipeline()

# ==================================================
# Decision Mapping
# ==================================================
decision_map = {
    0: "❌ HIGH RISK – DO NOT APPROVE",
    1: "✅ APPROVE LOAN"
}

# ==================================================
# Helper Functions
# ==================================================
def align_input_schema(input_df, pipeline):
    expected_cols = pipeline.feature_names_in_

    for col in expected_cols:
        if col not in input_df.columns:
            input_df[col] = np.nan

    return input_df[expected_cols]


def calculate_overall_missing_percentage(input_df, pipeline):
    expected_features = pipeline.feature_names_in_
    total_features = len(expected_features)

    missing_features = []

    for col in expected_features:
        if col not in input_df.columns:
            missing_features.append(col)
        elif input_df[col].isna().any():
            missing_features.append(col)

    missing_percent = (len(missing_features) / total_features) * 100
    return missing_percent, missing_features

# ==================================================
# Instructions
# ==================================================
with st.expander("📌 Instructions for Bank Employee", expanded=True):
    st.markdown("""
    • Upload **CSV or Excel** file  
    • Missing fields are allowed  
    • Missing model features are auto-filled  
    • Review data completeness before decision  
    • Final approval remains manual  
    """)

# ==================================================
# File Upload
# ==================================================
uploaded_file = st.file_uploader(
    "📥 Upload Customer File (CSV / Excel)",
    type=["csv", "xlsx"]
)

if uploaded_file:

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("📄 Uploaded Customer Data")
    st.dataframe(df, use_container_width=True)

    tab1, tab2, tab3 = st.tabs(
        ["📊 Data Completeness", "📋 Missing Features", "🤖 Loan Decision"]
    )

    # ---------------- TAB 1 ----------------
    with tab1:
        missing_percent, missing_features = calculate_overall_missing_percentage(df, pipeline)

        st.metric("Missing Data Percentage", f"{missing_percent:.2f}%")

        if missing_percent == 0:
            st.success("Excellent data quality")
        elif missing_percent < 30:
            st.info("Good data quality")
        elif missing_percent < 60:
            st.warning("Moderate data quality – review recommended")
        else:
            st.error("Poor data quality – low confidence")

    # ---------------- TAB 2 ----------------
    with tab2:
        if not missing_features:
            st.success("No model features missing")
        else:
            st.warning(f"{len(missing_features)} features auto-filled")
            st.dataframe(pd.DataFrame({"Missing Features": missing_features}))

    # ---------------- TAB 3 ----------------
    with tab3:
        try:
            aligned_df = align_input_schema(df.copy(), pipeline)

            preds = pipeline.predict(aligned_df)
            probs = pipeline.predict_proba(aligned_df)

            output_df = df.copy()
            output_df["Risk_Category"] = preds
            output_df["Loan_Decision"] = output_df["Risk_Category"].map(decision_map)

            st.dataframe(
                output_df[["Risk_Category", "Loan_Decision"]],
                use_container_width=True
            )

            if len(output_df) == 1:
                st.success(f"Final Decision: {output_df['Loan_Decision'].iloc[0]}")

            st.subheader("Risk Probability Breakdown")
            st.dataframe(pd.DataFrame(probs, columns=pipeline.classes_))

            st.download_button(
                "⬇ Download Decision Report",
                output_df.to_csv(index=False),
                "loan_decision_output.csv",
                "text/csv"
            )

        except Exception as e:
            st.error("Model could not process the uploaded file")
            st.code(str(e))

else:
    st.info("Please upload a CSV or Excel file to begin")

st.markdown("---")
st.caption("© Bank Credit Modelling System | Internal Use Only")
