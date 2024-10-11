import streamlit as st
import pandas as pd
import joblib

# Load model and preprocessor
model = joblib.load("models/best_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")

# Streamlit UI setup
st.title("Machine Learning Model Deployment with Streamlit")

# Input fields
st.write("Input the values for prediction:")

def get_user_input():
    """Create input fields dynamically based on the dataset columns."""
    # You can define your input fields manually or based on dataset columns
    feature1 = st.number_input("Feature 1", min_value=0.0, step=0.1)
    feature2 = st.number_input("Feature 2", min_value=0.0, step=0.1)
    feature3 = st.number_input("Feature 3", min_value=0.0, step=0.1)

    # Return data as a DataFrame
    return pd.DataFrame({
        "feature1": [feature1],
        "feature2": [feature2],
        "feature3": [feature3]
    })

# Get input from user
user_input = get_user_input()

# Button to make prediction
if st.button("Predict"):
    # Preprocess the user input
    X_new = preprocessor.transform(user_input)

    # Make prediction
    prediction = model.predict(X_new)

    # Show result
    st.write(f"Prediction: {prediction[0]}")

# To run the Streamlit app, use the command:
# streamlit run app_streamlit.py
