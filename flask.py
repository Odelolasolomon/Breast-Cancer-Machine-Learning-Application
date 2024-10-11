from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load model and preprocessor
model = joblib.load("models/best_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")

@app.route('/')
def home():
    return "ML Model API with Flask"

@app.route('/predict', methods=['POST'])
def predict():
    """
    Make predictions on incoming data via POST request.
    Data should be sent as JSON in the format {"feature1": value1, "feature2": value2, ...}.
    """
    # Get JSON data from request
    data = request.get_json(force=True)

    # Convert JSON to DataFrame
    df = pd.DataFrame([data])

    # Preprocess the data
    X_new = preprocessor.transform(df)

    # Make prediction
    prediction = model.predict(X_new)

    # Return prediction as JSON
    return jsonify({"prediction": prediction[0]})

if __name__ == "__main__":
    # Run the app
    app.run(host="0.0.0.0", port=5000, debug=True)
