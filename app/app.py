import streamlit as st
import joblib

# Load model
@st.cache_resource
def load_model():
    return {
        'model': joblib.load('models/kmeans_model.pkl'),
        'scaler': joblib.load('models/scaler.pkl')
    }

data = load_model()

st.title("Customer Segment Predictor")
income = st.number_input("Income", min_value=0)

if st.button("Predict"):
    prediction = data['model'].predict([[income]])[0]  # Add all your features
    st.success(f"Segment: {prediction}")
