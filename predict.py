"""
Prediction script for fake news detection
Use this script to predict whether a news article is fake or real
"""

import pickle
import re
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Configuration (must match training configuration)
MAX_SEQUENCE_LENGTH = 500

def preprocess_text(text):
    """Clean and preprocess text data (same as training)"""
    if not text:
        return ""
    
    text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def predict_news(text, model_path='best_lstm_model.h5', tokenizer_path='tokenizer.pkl'):
    """
    Predict if a news article is fake (0) or real (1)
    
    Args:
        text: News article text (string)
        model_path: Path to saved model (default: 'best_lstm_model.h5')
        tokenizer_path: Path to saved tokenizer (default: 'tokenizer.pkl')
    
    Returns:
        prediction: 'Fake' or 'Real'
        confidence: Confidence score (0-1)
    """
    try:
        # Load tokenizer
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        
        # Load model
        model = load_model(model_path)
        
        # Preprocess text
        cleaned_text = preprocess_text(text)
        
        # Tokenize and pad
        sequence = tokenizer.texts_to_sequences([cleaned_text])
        padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
        
        # Predict
        prediction_proba = model.predict(padded_sequence, verbose=0)[0][0]
        prediction = 'Real' if prediction_proba > 0.5 else 'Fake'
        confidence = max(prediction_proba, 1 - prediction_proba)
        
        return prediction, confidence
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure you have trained the model first and the files exist.")
        return None, None
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None

def main():
    """Interactive prediction"""
    print("="*60)
    print("Fake News Detection - Prediction Tool")
    print("="*60)
    print("\nEnter the news article text (or 'quit' to exit):\n")
    
    # Ask which model to use
    model_choice = input("Which model to use? (1) LSTM (2) GRU [default: 1]: ").strip()
    if model_choice == '2':
        model_path = 'best_gru_model.h5'
        print("Using GRU model")
    else:
        model_path = 'best_lstm_model.h5'
        print("Using LSTM model")
    
    while True:
        print("\n" + "-"*60)
        text = input("News article: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not text:
            print("Please enter some text.")
            continue
        
        prediction, confidence = predict_news(text, model_path)
        
        if prediction:
            print(f"\nPrediction: {prediction}")
            print(f"Confidence: {confidence:.2%}")
            print(f"Score: {confidence:.4f}")

if __name__ == "__main__":
    main()

