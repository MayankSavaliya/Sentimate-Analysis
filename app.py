import os
from flask import Flask, request, jsonify, render_template
from model_utils import load_model, predict_sentiment

app = Flask(__name__)

# Load the model when the application starts
model = None

@app.route('/', methods=['GET'])
def index():
    # Serve the frontend template
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global model
    
    # Load model if not already loaded
    if model is None:
        model = load_model()
        if model is None:
            return jsonify({'error': 'Failed to load model'}), 500
    
    try:
        # Get text from request
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON'}), 400
        
        text = data.get('text')
        if not text:
            return jsonify({'error': 'Missing "text" field'}), 400
        
        # Predict sentiment
        sentiment_label, confidence = predict_sentiment(model, text)
        
        return jsonify({
            'sentiment': sentiment_label,
            'confidence': float(confidence)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
