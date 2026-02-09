import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Page config (compact layout)
st.set_page_config(page_title="Network Intrusion Detection")

st.title("🔐 Network Intrusion Detection Dashboard")
st.write("Upload network flow CSV file to detect malicious activity.")

# Load model safely
try:
    model = joblib.load("models/random_forest_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
except:
    st.error("Model not found. Run train_model.py first.")
    st.stop()

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        # Clean column names
        data.columns = data.columns.str.strip()

        st.subheader("📄 Data Preview")
        st.dataframe(data.head())

        # Remove label if exists
        if "Label" in data.columns:
            data = data.drop("Label", axis=1)

        # Clean data for prediction
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.fillna(0, inplace=True)
        data = data.select_dtypes(include=[np.number])

        # Scale
        data_scaled = scaler.transform(data)

        # Predict
        predictions = model.predict(data_scaled)
        probabilities = model.predict_proba(data_scaled)[:, 1]

        data["Prediction"] = predictions
        data["Attack Probability"] = probabilities

        st.subheader("📊 Prediction Results")
        st.dataframe(data.head())

        attack_count = int(np.sum(predictions))
        normal_count = len(predictions) - attack_count
        total = len(predictions)
        attack_ratio = attack_count / total

        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", total)
        col2.metric("Normal Traffic", normal_count)
        col3.metric("Detected Attacks", attack_count)

        # -----------------------
        # Charts Section
        # -----------------------

        st.subheader("📈 Traffic Analysis")

        col4, col5 = st.columns(2)

        # Pie Chart
        with col4:
            fig1, ax1 = plt.subplots(figsize=(4, 4))
            ax1.pie(
                [normal_count, attack_count],
                labels=["Normal", "Attack"],
                autopct="%1.1f%%"
            )
            ax1.set_title("Attack Distribution")
            st.pyplot(fig1)

        # Bar Chart
        with col5:
            fig2, ax2 = plt.subplots(figsize=(5, 3))
            ax2.bar(["Normal", "Attack"], [normal_count, attack_count])
            ax2.set_title("Prediction Count")
            st.pyplot(fig2)

        # Probability Histogram
        st.subheader("📉 Attack Probability Distribution")

        fig3, ax3 = plt.subplots(figsize=(5, 3))
        ax3.hist(probabilities, bins=30)
        ax3.set_title("Probability Histogram")
        st.pyplot(fig3)

        # Feature Importance
        st.subheader("🔥 Top 10 Important Features")

        feature_names = data.drop(
            ["Prediction", "Attack Probability"], axis=1
        ).columns

        importances = model.feature_importances_
        feat_importance = pd.Series(importances, index=feature_names)
        top_features = feat_importance.nlargest(10)

        fig4, ax4 = plt.subplots(figsize=(6, 4))
        top_features.plot(kind="barh")
        ax4.set_title("Top Features")
        st.pyplot(fig4)

        # Alert Logic (Professional threshold)
        if attack_ratio > 0.05:
            st.error("⚠ High malicious activity detected!")
        else:
            st.success("✅ Network traffic mostly safe.")

    except Exception as e:
        st.error(f"Error: {e}")
