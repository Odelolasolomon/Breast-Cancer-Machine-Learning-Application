from fastapi import FastAPI
import pandas as pd
import joblib

# Initialize FastAPI app
app = FastAPI()

# Load model and preprocessor
model = joblib.load("models/best_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")

@app.get("/")
def home():
    return {"message": "ML Model API with FastAPI"}

@app.post("/predict/")
def predict(data: dict):
    """
    Make predictions on incoming data.
    Data should be provided as a dictionary with keys matching the features.
    """
    # Convert incoming JSON data to pandas DataFrame
    df = pd.DataFrame([data])

    # Preprocess the data
    X_new = preprocessor.transform(df)

    # Make prediction
    prediction = model.predict(X_new)

    # Return prediction as response
    return {"prediction": prediction[0]}

# To run the FastAPI app, use the following command:
# uvicorn app_fastapi:app --reload --host 0.0.0.0 --port 8000
