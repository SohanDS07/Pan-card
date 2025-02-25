# app.py (Final Updated Code)
import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np
import re
from werkzeug.utils import secure_filename
from functools import wraps
import time
import base64
import io
import uuid  # Import uuid for generating unique API keys

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'webp'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

reader = easyocr.Reader(['en'], gpu=False)

# --- API Key Management (Simplified In-Memory Dictionary) ---
API_KEYS = {}  # Start with an empty dictionary, keys will be generated and added here

def generate_api_key():
    """Generates a unique API key."""
    return str(uuid.uuid4())  # Using uuid for simple unique key generation

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key or api_key not in API_KEYS:
            return jsonify({"error": "Invalid or missing API key"}), 401
        return f(*args, **kwargs)
    return decorated_function

def is_black_and_white(image_path, threshold=10):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pixel_std_devs = [np.std(pixel) for row in img_rgb for pixel in row]
        avg_pixel_std_dev = np.mean(pixel_std_devs)

        return avg_pixel_std_dev < threshold
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def analyze_pan_card(image_path):
    is_bw = is_black_and_white(image_path)

    if is_bw is True:
        result_message = "Detected a black-and-white (Xerox) image. Not a PAN Card."
        return result_message, None, True, False, None # Added None for base64 image
    elif is_bw is None:
        result_message = "Could not determine if image is black and white. Proceeding with PAN card detection, but might not be accurate."
    elif is_bw is False:
        result_message = "Detected color image. Proceeding with PAN card detection."

    img = cv2.imread(image_path)
    img_for_display = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    text_ = reader.readtext(img)

    threshold = 0.25
    pan_keywords = ["INCOME TAX DEPARTMENT", "GOVT OF INDIA", "Permanent Account Number", "Signature"]
    pan_number_pattern = r'[A-Z]{5}[0-9]{4}[A-Z]{1}'

    keywords_found = set()
    pan_number_found = False

    for bbox, text, score in text_:
        if score > threshold:
            for keyword in pan_keywords:
                if keyword.lower() in text.lower():
                    print(f"Found PAN card related text: {text}")
                    keywords_found.add(keyword)
                    cv2.rectangle(img, bbox[0], bbox[2], (0, 255, 0), 5)
                    cv2.putText(img, text, bbox[0], cv2.FONT_HERSHEY_COMPLEX, 0.65, (255, 0, 0), 2)

            if re.fullmatch(pan_number_pattern, text):
                print(f"Found PAN number: {text}")
                pan_number_found = True
                cv2.rectangle(img, bbox[0], bbox[2], (0, 255, 0), 5)
                cv2.putText(img, text, bbox[0], cv2.FONT_HERSHEY_COMPLEX, 0.65, (255, 0, 0), 2)

    if len(keywords_found) >= 3 or (len(keywords_found) >= 2 and pan_number_found):
        result_message = "This is a PAN card."
        is_pan = True
    else:
        result_message = "Not a PAN card."
        is_pan = False

    # --- Encode annotated image to base64 ---
    _, img_encoded = cv2.imencode('.jpg', img) # Encode to JPEG format
    img_base64 = base64.b64encode(img_encoded).decode('utf-8') # Base64 encode and decode to string

    return result_message, None, False, is_pan, img_base64 # Return base64 image, no filename for API


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    print("Function upload_file is called")
    if request.method == 'POST':
        print("POST request received at /")
        if 'file' not in request.files:
            print("No file part")
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            print("No selected file")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            print(f"File saved: {filepath}")
            return redirect(url_for('classify_image', filename=filename))
    else:
        print("GET request received at /")
    return render_template('index.html', result=None, image_url=None, progress=0)

@app.route('/classify/<filename>')
def classify_image(filename):
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    start_time = time.time()
    result_message, annotated_image_name, is_bw_image, _, _ = analyze_pan_card(image_path) # Ignore base64 and is_pan for web UI
    end_time = time.time()
    processing_time = end_time - start_time
    progress_percentage = 100

    annotated_image_url = None
    if annotated_image_name:
        annotated_image_url = url_for('uploaded_file', filename=annotated_image_name)

    original_image_url = url_for('uploaded_file', filename=filename)

    return render_template(
        'index.html',
        result=result_message,
        image_url=original_image_url,
        annotated_image_url=annotated_image_url,
        progress=progress_percentage,
        is_bw=is_bw_image
    )

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# --- API Endpoint to Get a New API Key ---
@app.route('/api/get_api_key', methods=['GET'])
def get_api_key():
    """Endpoint to generate and provide a new API key."""
    new_key = generate_api_key()
    API_KEYS[new_key] = "user_placeholder"  # In real app, associate with user info
    print(f"Generated API Key: {new_key}") # Log the generated key (for demonstration)
    return jsonify({"api_key": new_key, "message": "API key generated successfully. Please use this key in the X-API-Key header."}), 200


# --- API Endpoint for Classification (No Changes) ---
@app.route('/api/classify_pan_card', methods=['POST'])
@require_api_key
def api_classify_pan_card():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400

    image_file = request.files['image']

    if image_file.filename == '':
        return jsonify({'error': 'No selected image'}), 400

    if image_file and allowed_file(image_file.filename):
        filename = secure_filename(image_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(filepath)
        result_message, _, _, is_pan_card, annotated_image_base64 = analyze_pan_card(filepath) # Get base64 image
        os.remove(filepath)

        return jsonify({
            'is_pan_card': is_pan_card,
            'message': result_message,
            'annotated_image_base64': annotated_image_base64 # Include base64 image in response
        }), 200
    else:
        return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True)