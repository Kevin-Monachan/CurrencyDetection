from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from model import load_model_and_predict

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load your trained model and classes
MODEL_PATH = 'D:\IndianCurrencyDetectioModel/best_resnext50_indian_currency_model.pth' # Make sure this path is correct
MODEL_NAME = 'resnext50'
CLASS_NAMES = ['10', '20', '50', '100', '200', '500'] # Must match your training classes

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess and get prediction from the model
        predicted_denomination, confidence = load_model_and_predict(
            filepath, MODEL_PATH, MODEL_NAME, CLASS_NAMES
        )
        
        # Clean up the temporary file
        os.remove(filepath)
        
        return jsonify({
            'prediction': predicted_denomination,
            'confidence': f"{confidence:.2f}"
        })

if __name__ == '__main__':
    app.run(debug=True)