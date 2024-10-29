from flask import Flask, request, jsonify
import numpy as np
import joblib
import tensorflow as tf
import os

app = Flask(__name__)

# Load the TFLite model
model_path = os.path.join(os.path.dirname(__file__), '../test.tflite')
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

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

    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set the tensor for the input data
    interpreter.set_tensor(input_details[0]['index'], inputs_scaled.astype(np.float32))

    # Run inference
    interpreter.invoke()

    # Get the prediction from the output tensor
    prediction = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(prediction, axis=1)[0]

    return jsonify({"prediction": int(predicted_class)})

# Vercel will automatically handle the server, so no app.run() is needed
