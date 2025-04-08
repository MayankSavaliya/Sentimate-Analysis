import os
from flask import Flask, request, jsonify, render_template
from model_utils import load_model, predict_sentiment, unload_model
import gc
import threading
import time

app = Flask(__name__)

# Set environment variables for lower memory usage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU

# Global variables
model = None
model_lock = threading.Lock()
last_used_time = None
MODEL_UNLOAD_TIMEOUT = 300  # Unload model after 5 minutes of inactivity

@app.route('/', methods=['GET'])
def index():
    # Serve the frontend template
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global model, last_used_time
    
    try:
        # Get text from request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON'}), 400
        
        text = data.get('text')
        if not text:
            return jsonify({'error': 'Missing "text" field'}), 400
        
        # Load model if not already loaded (thread-safe)
        with model_lock:
            if model is None:
                app.logger.info("Loading model...")
                model = load_model()
                if model is None:
                    return jsonify({'error': 'Failed to load model'}), 500
            
            # Update last used time
            last_used_time = time.time()
            
            # Predict sentiment
            sentiment_label, confidence = predict_sentiment(model, text)
        
        # Force garbage collection
        gc.collect()
        
        return jsonify({
            'sentiment': sentiment_label,
            'confidence': float(confidence)
        })
        
    except Exception as e:
        app.logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({'status': 'ok'})

def check_model_timeout():
    """Background thread to unload model after inactivity"""
    global model, last_used_time
    
    while True:
        time.sleep(60)  # Check every minute
        
        if model is not None and last_used_time is not None:
            if time.time() - last_used_time > MODEL_UNLOAD_TIMEOUT:
                app.logger.info("Unloading model due to inactivity...")
                with model_lock:
                    unload_model()
                    model = None

if __name__ == '__main__':
    # Start background thread for model timeout
    timeout_thread = threading.Thread(target=check_model_timeout, daemon=True)
    timeout_thread.start()
    
    # Use threaded server with limited workers
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), 
            threaded=True)
