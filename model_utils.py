import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_model():
    """Load the trained model, tokenizer and metadata"""
    try:
        # Define file paths
        model_path = os.path.join('models', 'saved_model', 'model.keras')
        tokenizer_path = os.path.join('models', 'tokenizer.pickle')
        metadata_path = os.path.join('models', 'model_metadata.pickle')
        
        # Debugging: Print current working directory and directory contents
        print(f"Current working directory: {os.getcwd()}")
        print(f"Directory contents: {os.listdir('.')}")
        if os.path.exists('models'):
            print(f"Models directory contents: {os.listdir('models')}")
            if os.path.exists(os.path.join('models', 'saved_model')):
                print(f"Saved model directory contents: {os.listdir(os.path.join('models', 'saved_model'))}")
        
        # Check if files exist
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return None
            
        if not os.path.exists(tokenizer_path):
            print(f"Error: Tokenizer file not found at {tokenizer_path}")
            return None
            
        if not os.path.exists(metadata_path):
            print(f"Error: Metadata file not found at {metadata_path}")
            return None
        
        # Load model
        model = tf.keras.models.load_model(model_path)
        
        # Load tokenizer
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        
        # Load metadata
        with open(metadata_path, 'rb') as handle:
            metadata = pickle.load(handle)
        
        # Store tokenizer and metadata with the model for easy access
        model.tokenizer = tokenizer
        model.metadata = metadata
        
        print("Model, tokenizer and metadata loaded successfully")
        return model
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def predict_sentiment(model, text):
    """
    Predict sentiment for the given text using the loaded model
    Returns the sentiment label and confidence score
    """
    try:
        # Access tokenizer and metadata from model
        tokenizer = model.tokenizer
        metadata = model.metadata
        max_length = metadata['max_length']
        index_to_label = metadata['index_to_label']
        
        # Preprocess text
        sequences = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequences, maxlen=max_length)
        
        # Make prediction
        prediction = model.predict(padded, verbose=0)[0]
        predicted_class = np.argmax(prediction)
        confidence = prediction[predicted_class]
        
        # Map index to label
        sentiment_label = index_to_label.get(int(predicted_class), "Unknown")
        
        return sentiment_label, confidence
    
    except Exception as e:
        print(f"Error predicting sentiment: {str(e)}")
        return "Error", 0.0
