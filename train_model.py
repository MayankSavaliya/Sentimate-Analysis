import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set environment variable for Keras backend
os.environ["KERAS_BACKEND"] = "tensorflow"

# Set random seed for reproducibility
tf.random.set_seed(97)
np.random.seed(97)

def setup_directories():
    """Create necessary directories if they don't exist"""
    dirs = ['data', 'models', 'models/saved_model']
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
    print("Directory structure created.")

def load_data():
    """Load and prepare the dataset"""
    file_path = os.path.join('data', 'Combined Data.csv')
    df = pd.read_csv(file_path)
    
    # Drop unnamed column if it exists
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    # Rename 'Mental Health Status' column to 'status' if it exists
    if 'Mental Health Status' in df.columns:
        df = df.rename(columns={'Mental Health Status': 'status'})
    
    # Drop rows with missing or invalid data
    df.dropna(subset=['statement', 'status'], inplace=True)
    
    # Extract unique status values
    unique_statuses = sorted(df['status'].unique())
    print(f"Unique status values in dataset: {unique_statuses}")
    
    # Create label mapping dynamically
    SENTIMENT_LABELS = {status: i for i, status in enumerate(unique_statuses)}
    print(f"Created label mapping: {SENTIMENT_LABELS}")
    
    # Reverse mapping for predictions
    INDEX_TO_LABEL = {v: k for k, v in SENTIMENT_LABELS.items()}
    
    # Convert status labels to numeric indices
    df['sentiment_index'] = df['status'].map(SENTIMENT_LABELS)
    
    print(f"Dataset loaded. Shape: {df.shape}")
    return df, SENTIMENT_LABELS, INDEX_TO_LABEL

def create_tokenizer(texts, max_words=10000):
    """Create and fit a tokenizer on the texts"""
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    
    # Save tokenizer for later use in prediction
    with open('models/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return tokenizer

def preprocess_texts(texts, tokenizer, max_length=100):
    """Convert texts to padded sequences"""
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=max_length)

def create_model(vocab_size, num_classes, embedding_dim=128, max_length=100):
    """Create a simple but effective model for sentiment analysis"""
    print(f"Creating model with {num_classes} output classes")
    
    model = keras.Sequential([
        keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)),
        keras.layers.Bidirectional(keras.layers.LSTM(32)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation='softmax')  # Classes based on actual dataset
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(df, INDEX_TO_LABEL):
    """Train the model with the prepared data"""
    print("Preparing data...")
    
    # Get texts and labels
    texts = df['statement'].values
    labels = df['sentiment_index'].values
    
    # Create and fit tokenizer
    print("Creating tokenizer...")
    tokenizer = create_tokenizer(texts)
    
    # Get vocabulary size
    vocab_size = len(tokenizer.word_index) + 1
    print(f"Vocabulary size: {vocab_size}")
    
    # Preprocess texts
    print("Preprocessing texts...")
    X = preprocess_texts(texts, tokenizer)
    
    # Split data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=101)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=101)
    
    # Get number of classes
    num_classes = len(np.unique(labels))
    
    # Create model
    print("Creating model...")
    model = create_model(vocab_size, num_classes)
    model.summary()
    
    # Early stopping callback
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=2,
        mode='min',
        restore_best_weights=True
    )
    
    # Train model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,
        batch_size=32,
        callbacks=[early_stopping]
    )
    
    # Evaluate model
    print("Evaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Save metadata for predictions
    metadata = {
        'max_length': X.shape[1],
        'index_to_label': INDEX_TO_LABEL
    }
    
    with open('models/model_metadata.pickle', 'wb') as handle:
        pickle.dump(metadata, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return model, history

if __name__ == "__main__":
    print("Starting the training process...")
    
    # Setup directories
    setup_directories()
    
    # Load data and extract labels
    df, SENTIMENT_LABELS, INDEX_TO_LABEL = load_data()
    if df is None:
        print("Failed to load data. Exiting.")
        sys.exit(1)
    
    # Train model
    model, history = train_model(df, INDEX_TO_LABEL)
    
    # Save model
    print("Saving model...")
    save_dir = os.path.join('models', 'saved_model')
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, 'model.keras')
    model.save(model_path)
    
    print(f"Model saved to {model_path}")
    print("Training completed successfully!")
