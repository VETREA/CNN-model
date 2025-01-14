from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import os
from PIL import Image


app = Flask(__name__)

model_path = os.path.join(os.path.dirname(__file__), 'leaf_disease_model.h5')

model = load_model(model_path)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

IMAGE_SIZE = (256, 256)
CLASS_NAMES = ['applescab', 'blackrot', 'healthy']  

def prepare_image(image, target_size):
    """Preprocess the image to the required size and format."""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255.0  
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        image = Image.open(file.stream)  
        prepared_image = prepare_image(image, IMAGE_SIZE)
        prediction = model.predict(prepared_image)
        predicted_class = CLASS_NAMES[np.argmax(prediction)]
        confidence = np.max(prediction)

        return jsonify({
            'predicted_class': predicted_class,
            'confidence': float(confidence)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
