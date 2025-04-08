import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gc

# Global cache to prevent reloading model repeatedly
_model_cache = None
_tokenizer_cache = None
_metadata_cache = None

def load_model():
    """
    Load the saved model and required files.
    Implements memory optimization for deployment.
    """
    global _model_cache, _tokenizer_cache, _metadata_cache
    
    # Return cached model if available
    if _model_cache is not None:
        return _model_cache
    
    try:
        # Configure TensorFlow for memory optimization
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            # Restrict TensorFlow to only use a portion of GPU memory
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        
        # Set memory limits
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=512)]
        ) if gpus else None
        
        # Load model with memory optimization
        model_path = os.path.join('models', 'saved_model', 'model.keras')
        
        # Load model with reduced precision
        _model_cache = keras.models.load_model(
            model_path, 
            compile=False  # Don't compile to save memory
        )
        
        # Compile with minimal metrics
        _model_cache.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Load tokenizer
        with open('models/tokenizer.pickle', 'rb') as handle:
            _tokenizer_cache = pickle.load(handle)
        
        # Load metadata
        with open('models/model_metadata.pickle', 'rb') as handle:
            _metadata_cache = pickle.load(handle)
        
        # Force garbage collection
        gc.collect()
        
        return _model_cache
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def predict_sentiment(model, text):
    """
    Predict sentiment for given text.
    Uses memory-efficient processing.
    """
    global _tokenizer_cache, _metadata_cache
    
    try:
        # Check if tokenizer and metadata are available
        if _tokenizer_cache is None or _metadata_cache is None:
            load_model()  # This will populate the caches
        
        # Process input text
        sequences = _tokenizer_cache.texts_to_sequences([text])
        max_length = _metadata_cache['max_length']
        padded_sequences = pad_sequences(sequences, maxlen=max_length)
        
        # Make prediction with smaller batch size
        prediction = model.predict(padded_sequences, batch_size=1, verbose=0)
        
        # Get sentiment label
        predicted_index = np.argmax(prediction, axis=1)[0]
        confidence = float(prediction[0][predicted_index])
        sentiment_label = _metadata_cache['index_to_label'][predicted_index]
        
        # Force garbage collection
        gc.collect()
        
        return sentiment_label, confidence
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return "Error", 0.0

def unload_model():
    """
    Explicitly unload model to free memory
    """
    global _model_cache, _tokenizer_cache, _metadata_cache
    
    _model_cache = None
    _tokenizer_cache = None
    _metadata_cache = None
    
    # Force garbage collection
    gc.collect()
