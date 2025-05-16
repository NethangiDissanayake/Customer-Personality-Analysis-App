import streamlit as st
import joblib
import numpy as np

# Load model
try:
    model_data = joblib.load('models/kmeans_model.pkl')
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Input fields for all 12 features
st.header("Customer Segmentation Predictor")

col1, col2 = st.columns(2)
with col1:
    income = st.number_input("Income ($)", min_value=0)
    kidhome = st.number_input("Number of kids at home", min_value=0, max_value=5)
    teenhome = st.number_input("Number of teens at home", min_value=0, max_value=5)
    recency = st.number_input("Days since last purchase", min_value=0)
    mnt_wines = st.number_input("Annual wine spending ($)", min_value=0)
    mnt_fruits = st.number_input("Annual fruit spending ($)", min_value=0)

with col2:
    mnt_meat = st.number_input("Annual meat spending ($)", min_value=0)
    mnt_fish = st.number_input("Annual fish spending ($)", min_value=0)
    mnt_sweets = st.number_input("Annual sweets spending ($)", min_value=0)
    mnt_gold = st.number_input("Annual gold products spending ($)", min_value=0)
    deals = st.number_input("Discounted purchases count", min_value=0)
    web_purchases = st.number_input("Online purchases count", min_value=0)

# Prediction button
if st.button("Predict Segment"):
    # Create array with ALL 12 features in EXACT order
    input_data = np.array([
        income, kidhome, teenhome, recency,
        mnt_wines, mnt_fruits, mnt_meat,
        mnt_fish, mnt_sweets, mnt_gold,
        deals, web_purchases
    ]).reshape(1, -1)  # Reshape to (1, 12)
    
    # Debug: Print shape and values
    st.write("Input data shape:", input_data.shape)
    st.write("Input values:", input_data)
    
    try:
        prediction = model_data['model'].predict(input_data)[0]
        st.success(f"This customer belongs to segment: {prediction}")
    except Exception as e:
        st.error(f"Prediction error: {e}")

