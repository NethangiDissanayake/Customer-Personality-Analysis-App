import streamlit as st
import joblib
import numpy as np

# Load model with enhanced verification
try:
    model_data = joblib.load('models/kmeans_model.pkl')
    model = model_data['model']  # Directly access the model
    
    # Verify model is properly trained
    if not hasattr(model, 'cluster_centers_'):
        st.error("Model loaded but not trained! Please retrain the model.")
        st.stop()
        
    st.success("✅ Model loaded and verified successfully!")
    
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Prediction function
def predict_segment(features):
    """Takes list of 12 feature values, returns segment"""
    try:
        return model.predict(np.array(features).reshape(1, -1))[0]
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None

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
    input_features = [
        income, kidhome, teenhome, recency,
        mnt_wines, mnt_fruits, mnt_meat,
        mnt_fish, mnt_sweets, mnt_gold,
        deals, web_purchases
    ]
    
    # Debug info
    with st.expander("Debug Info"):
        st.write("Input features:", input_features)
        st.write("Feature count:", len(input_features))
    
    # Make prediction
    segment = predict_segment(input_features)
    
    if segment is not None:
        st.success(f"## Predicted Customer Segment: {segment}")
        
        # Optional: Show cluster characteristics
        with st.expander("Cluster Details"):
            st.write("Cluster center values:", model.cluster_centers_[segment])
