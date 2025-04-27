from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Initialize Flask app
app = Flask(__name__)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="final.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define your class names (must match the training order)
class_names = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']  # Example classes

# Routes
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', prediction="No file uploaded!")

    file = request.files['image']

    if file.filename == '':
        return render_template('index.html', prediction="No selected file!")

    if file:
        try:
            # Preprocess the image
            img = Image.open(file.stream).convert('RGB')
            img = img.resize((64, 64))  # Match training size
            img = np.array(img, dtype=np.float32) / 255.0
            img = np.expand_dims(img, axis=0)

            # Predict
            interpreter.set_tensor(input_details[0]['index'], img)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])

            prediction_idx = np.argmax(output_data)
            prediction_label = class_names[prediction_idx]
            confidence = np.max(output_data) * 100  # Get highest probability

            # Send both label and confidence to frontend
            return render_template('index.html', 
                                   prediction=prediction_label, 
                                   confidence=f"{confidence:.2f}%")

        except Exception as e:
            return render_template('index.html', prediction=f"Error: {str(e)}")

    return render_template('index.html', prediction="Unknown error occurred!")

# Run app
if __name__ == '__main__':
    app.run(debug=True)
