import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Churn Prediction Demo", layout="wide")
st.title("📊 Churn Prediction Interactive Demo")
st.write("Made By Subhadip")

# Sidebar - Upload
uploaded_file = st.sidebar.file_uploader("Upload XGBoost prediction CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Decode if needed
    geography_map = {0: "France", 1: "Germany", 2: "Spain"}
    gender_map = {0: "Female", 1: "Male"}
    if "Geography" in df.columns:
        df["Geography"] = df["Geography"].map(geography_map)
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map(gender_map)

    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    st.subheader("🎯 Churn Prediction Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x="Predicted_Churn", ax=ax1)
    ax1.set_xticklabels(["No Churn", "Churn"])
    st.pyplot(fig1)

    st.subheader("📈 Churn Probability Histogram")
    fig2, ax2 = plt.subplots()
    sns.histplot(df["Churn_Probability"], bins=30, kde=True, ax=ax2)
    st.pyplot(fig2)

    st.subheader("⚠️ High-Risk Customers")
    threshold = st.slider("Churn probability threshold", 0.0, 1.0, 0.5)
    high_risk = df[df["Churn_Probability"] >= threshold]
    st.write(f"Showing {len(high_risk)} high-risk customers (Prob ≥ {threshold})")
    st.dataframe(high_risk.reset_index(drop=True))

else:
    st.info("Please upload your XGBoost churn prediction CSV file.")
