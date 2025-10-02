# --- app.py (FLASK BACKEND for ResNet Model) ---

import tensorflow as tf
from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import io 
import re 
import os # To check for model file existence

# --- 1. Model Configuration and Loading ---
# The path to your model and label files
MODEL_PATH = 'D:\CurrencyDetection\models\indian_currency_resnet_final.keras'
LABEL_PATH = 'D:\CurrencyDetection\models\labels.txt'

# Configure Flask to look for index.html in the same directory (or a 'templates' folder)
app = Flask(__name__, template_folder='.') 

model = None
class_labels = []

def load_model_assets():
    """Loads the Keras model and labels on server startup."""
    global model, class_labels
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ FATAL ERROR: Model file not found at {MODEL_PATH}")
        return

    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False) 
        
        # --- FIX APPLIED HERE: Simplifies reading the labels.txt file ---
        with open(LABEL_PATH, 'r') as f:
            # Reads lines, strips whitespace, and ensures line is not empty
            class_labels = [line.strip() for line in f if line.strip()]
            
        print("✅ ResNet Model and Labels loaded successfully.")
        print(f"Loaded Labels: {class_labels}")

    except Exception as e:
        print(f"❌ FATAL ERROR: Failed to load ML Model or Labels: {e}")
        model = None
        class_labels = ["Error_Loading_Model"]

load_model_assets() # Execute this function when the server starts

# --------------------------------------------------------------------------
## 2. Image Preprocessing Function
# --------------------------------------------------------------------------

def preprocess_image(image_file):
    """
    Resizes the input image and applies the official ResNet50 preprocessing.
    """
    target_size = (224, 224) 
    
    # 1. Open image from in-memory bytes and convert to RGB
    img = Image.open(io.BytesIO(image_file.read())).convert('RGB')
    img = img.resize(target_size)
    
    # 2. Convert to NumPy array
    img_array = np.asarray(img)
    
    # 3. Add a batch dimension (1, 224, 224, 3)
    img_batch = np.expand_dims(img_array, axis=0)
    
    # 4. Apply the exact same preprocessing function used in Keras training
    processed_image = preprocess_input(img_batch)
    
    return processed_image
# --------------------------------------------------------------------------
## 3. Flask Routes
# --------------------------------------------------------------------------

@app.route('/')
def home():
    """Serves the index.html file."""
    return render_template('index.html') 

@app.route('/predict', methods=['POST'])
def predict():
    """Receives image, runs inference, and returns JSON result."""
    if model is None or not class_labels:
        return jsonify({'error': 'ML Model is not loaded on the server. Check server logs.'}), 503
        
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400

    image_file = request.files['image']
    
    try:
        # 1. Preprocess
        processed_image = preprocess_image(image_file)
        
        # 2. Predict
        # Use predict(image) instead of predict(image)[0] to avoid potential list index error
        prediction = model.predict(processed_image) 
        
        # 3. Post-process
        predicted_class_index = np.argmax(prediction)
        confidence = float(prediction[0][predicted_class_index]) * 100
        predicted_label = class_labels[predicted_class_index]

        # Extract numerical value (0, 10, 20, 50, etc.)
        predicted_value = 0
        match = re.search(r'\d+', predicted_label)
        if match:
            predicted_value = int(match.group(0))

        # 4. Return the results
        return jsonify({
            'prediction_label': predicted_label,
            'confidence': f"{confidence:.2f}",
            'value': predicted_value
        })

    except Exception as e:
        print(f"Prediction failed: {e}")
        return jsonify({'error': f'An error occurred during prediction: {str(e)}'}), 500

if __name__ == '__main__':
    # You must have installed: pip install Flask tensorflow numpy pillow
    print("\n--- Starting Flask Server ---\n")
    app.run(debug=True)