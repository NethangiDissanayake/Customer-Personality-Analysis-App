import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

# Load the pre-trained model and scaler
kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")

# Set up the page
st.set_page_config(page_title="Customer Personality Predictor", layout="centered")
st.title("🧠 Predict Customer Personality Cluster")
st.markdown("Fill in the details below to predict which customer segment they belong to.")

# Example input fields
age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Annual Income (k$)", min_value=0, max_value=200, value=60)
spending_score = st.slider("Spending Score (1-100)", 1, 100, 50)
children = st.slider("Number of Children", 0, 5, 1)
recency = st.number_input("Recency (days since last purchase)", min_value=0, max_value=100, value=30)

# Input features as a dataframe
input_data = pd.DataFrame([[age, income, spending_score, children, recency]],
                          columns=["Age", "Income", "SpendingScore", "Children", "Recency"])

# Predict cluster
if st.button("Predict Customer Cluster"):
    X_scaled = scaler.transform(input_data)
    cluster = kmeans.predict(X_scaled)[0]
    st.success(f"🏷️ This customer belongs to Cluster {cluster}")
