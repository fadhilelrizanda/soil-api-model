from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model(os.path.join(os.path.dirname(__file__), '../test.h5'))

# Load the scaler
scaler = joblib.load(os.path.join(os.path.dirname(__file__), '../scaler.joblib'))

@app.route('/predict', methods=['POST'])
def predict():
    # Extract individual variables from the JSON request
    data = request.json
    s1 = data.get("s1")
    s2 = data.get("s2")

    # Combine inputs into a NumPy array and apply scaling
    inputs = np.array([[s1, s2]])
    inputs_scaled = scaler.transform(inputs)

    # Run the prediction with the scaled inputs
    prediction = model.predict(inputs_scaled)
    predicted_class = np.argmax(prediction, axis=1)[0]

    return jsonify({"prediction": int(predicted_class)})

# Vercel will automatically handle the server, so no app.run() is needed
