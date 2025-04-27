from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# Disable TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Define blood groups
blood_groups = {
    0: "A+", 1: "AB-", 2: "A-", 3: "B+", 4: "B-", 5: "AB+", 6: "O-", 7: "O+"
}

# Initialize Flask app
app = Flask(__name__)

# Load model
model = tf.keras.models.load_model('final.h5')

# Preprocess uploaded image
def preprocess_image(img):
    img = img.resize((64, 64))
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Homepage
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        img = Image.open(io.BytesIO(file.read()))
        img = img.convert('RGB')
        img_array = preprocess_image(img)

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        predicted_blood_group = blood_groups.get(predicted_class, "Unknown")

        return render_template('result.html', prediction=predicted_blood_group)

    except Exception as e:
        return jsonify({'error': f'Error during prediction: {str(e)}'}), 500

# Run locally
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)