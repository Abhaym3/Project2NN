"""
Fake News Detection using LSTM/GRU Neural Network
This script implements a deep learning model to classify fake vs real news articles.
"""

import pandas as pd
import numpy as np
import re
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# Set random seeds for reproducibility
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# FORCE CPU ONLY - Configure before importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow messages
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU completely
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit'

np.random.seed(42)

# Import TensorFlow
import tensorflow as tf

# Force CPU device placement
with tf.device('/CPU:0'):
    # Configure TensorFlow to use CPU only
    try:
        # Hide all GPU devices
        physical_devices = tf.config.list_physical_devices('GPU')
        for device in physical_devices:
            tf.config.set_visible_devices([], 'GPU')
    except:
        pass
    
    # Configure CPU threading
    try:
        tf.config.threading.set_inter_op_parallelism_threads(0)  # Use all available threads
        tf.config.threading.set_intra_op_parallelism_threads(0)  # Use all available threads
    except:
        pass

tf.random.set_seed(42)

# Verify CPU is being used
print("="*60)
print("FORCING CPU-ONLY EXECUTION")
print("="*60)
print(f"TensorFlow version: {tf.__version__}")
print(f"Available devices: {[d.name for d in tf.config.list_physical_devices()]}")
print(f"CPU devices: {[d.name for d in tf.config.list_physical_devices('CPU')]}")
if len(tf.config.list_physical_devices('GPU')) > 0:
    print("WARNING: GPU detected but will be disabled")
else:
    print("✓ No GPU detected - using CPU only")
print("="*60)

# Configuration
MAX_SEQUENCE_LENGTH = 500
MAX_NB_WORDS = 10000
EMBEDDING_DIM = 128
LSTM_UNITS = 128
GRU_UNITS = 128
BATCH_SIZE = 64
EPOCHS = 20
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1

def load_data():
    """Load and combine fake and true news datasets"""
    print("\n" + "="*60)
    print("STEP 1: Loading datasets...")
    print("="*60)
    
    # Load fake news
    print("Loading Fake.csv...")
    fake_df = pd.read_csv('Fake.csv')
    fake_df['label'] = 0  # 0 for fake
    print(f"  ✓ Loaded {len(fake_df)} fake news articles")
    
    # Load true news
    print("Loading True.csv...")
    true_df = pd.read_csv('True.csv')
    true_df['label'] = 1  # 1 for real
    print(f"  ✓ Loaded {len(true_df)} real news articles")
    
    # Combine datasets
    print("Combining datasets...")
    df = pd.concat([fake_df, true_df], ignore_index=True)
    
    # Shuffle the dataframe
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nTotal samples: {len(df)}")
    print(f"  - Fake news: {len(df[df['label'] == 0])}")
    print(f"  - Real news: {len(df[df['label'] == 1])}")
    
    return df

def preprocess_text(text):
    """Clean and preprocess text data"""
    if pd.isna(text):
        return ""
    
    # Convert to string
    text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters and digits (keep only letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def combine_title_text(row):
    """Combine title and text columns"""
    title = preprocess_text(row.get('title', ''))
    text = preprocess_text(row.get('text', ''))
    combined = title + ' ' + text
    return combined.strip()

def prepare_data(df):
    """Prepare text data for model training"""
    print("\n" + "="*60)
    print("STEP 2: Preprocessing text data...")
    print("="*60)
    
    # Combine title and text
    print("Combining title and text columns...")
    df['combined_text'] = df.apply(combine_title_text, axis=1)
    
    # Remove empty texts
    initial_count = len(df)
    df = df[df['combined_text'].str.len() > 0]
    removed = initial_count - len(df)
    
    if removed > 0:
        print(f"  ✓ Removed {removed} empty texts")
    
    # Get texts and labels
    texts = df['combined_text'].values
    labels = df['label'].values
    
    print(f"  ✓ Final dataset: {len(texts)} samples")
    
    return texts, labels

def tokenize_and_pad(texts, tokenizer=None, fit=True):
    """Tokenize and pad sequences"""
    if fit:
        print("\n" + "="*60)
        print("STEP 3: Tokenizing and padding sequences...")
        print("="*60)
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS, oov_token='<OOV>')
        print("Fitting tokenizer on texts...")
        tokenizer.fit_on_texts(texts)
        vocab_size = len(tokenizer.word_index) + 1
        print(f"  ✓ Vocabulary size: {vocab_size:,} words")
    else:
        print("Tokenizing texts...")
    
    print("Converting texts to sequences...")
    sequences = tokenizer.texts_to_sequences(texts)
    print(f"Padding sequences to length {MAX_SEQUENCE_LENGTH}...")
    padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    print(f"  ✓ Created {len(padded_sequences)} sequences")
    
    return padded_sequences, tokenizer

def build_lstm_model(vocab_size):
    """Build LSTM-based model"""
    print("Building LSTM model...")
    
    # Force CPU device for model building
    with tf.device('/CPU:0'):
        model = Sequential([
            Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH),
            Dropout(0.3),
            Bidirectional(LSTM(LSTM_UNITS, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(LSTM_UNITS // 2)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
    
    return model

def build_gru_model(vocab_size):
    """Build GRU-based model"""
    print("Building GRU model...")
    
    # Force CPU device for model building
    with tf.device('/CPU:0'):
        model = Sequential([
            Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH),
            Dropout(0.3),
            Bidirectional(GRU(GRU_UNITS, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(GRU(GRU_UNITS // 2)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
    
    return model

def plot_training_history(history):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision
    axes[1, 0].plot(history.history['precision'], label='Train Precision')
    axes[1, 0].plot(history.history['val_precision'], label='Val Precision')
    axes[1, 0].set_title('Model Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Recall
    axes[1, 1].plot(history.history['recall'], label='Train Recall')
    axes[1, 1].plot(history.history['val_recall'], label='Val Recall')
    axes[1, 1].set_title('Model Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("Training history plot saved as 'training_history.png'")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Fake', 'Real'], 
                yticklabels=['Fake', 'Real'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'confusion_matrix_{model_name.lower()}.png', dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved as 'confusion_matrix_{model_name.lower()}.png'")
    plt.close()

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model and print metrics"""
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name} Model")
    print(f"{'='*60}")
    
    # Predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\nTest Set Metrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, model_name)
    
    return accuracy, precision, recall, f1

def main():
    """Main function to run the fake news detection pipeline"""
    
    # Load data
    df = load_data()
    
    # Prepare data
    texts, labels = prepare_data(df)
    
    # Split data: 80% train, 10% validation, 10% test
    print("\n" + "="*60)
    print("STEP 4: Splitting data (80/10/10)...")
    print("="*60)
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, labels, test_size=TEST_SPLIT, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VALIDATION_SPLIT/(1-TEST_SPLIT), random_state=42, stratify=y_temp
    )
    
    print(f"  ✓ Training samples:   {len(X_train):,} (80%)")
    print(f"  ✓ Validation samples:  {len(X_val):,} (10%)")
    print(f"  ✓ Test samples:        {len(X_test):,} (10%)")
    
    # Tokenize and pad
    X_train_padded, tokenizer = tokenize_and_pad(X_train, fit=True)
    X_val_padded, _ = tokenize_and_pad(X_val, tokenizer=tokenizer, fit=False)
    X_test_padded, _ = tokenize_and_pad(X_test, tokenizer=tokenizer, fit=False)
    
    vocab_size = len(tokenizer.word_index) + 1
    print(f"\n  ✓ Final vocabulary size: {vocab_size:,}")
    
    # Save tokenizer
    print("\nSaving tokenizer...")
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    print("  ✓ Tokenizer saved as 'tokenizer.pkl'")
    
    # Convert labels to numpy arrays
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)
    
    # Build and train LSTM model
    print("\n" + "="*60)
    print("STEP 5: Building and Training LSTM Model")
    print("="*60)
    
    lstm_model = build_lstm_model(vocab_size)
    print("\nModel Architecture:")
    lstm_model.summary()
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
    
    model_checkpoint = ModelCheckpoint(
        'best_lstm_model.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=0.00001,
        verbose=1
    )
    
    # Train LSTM model
    print(f"\nStarting training...")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Max epochs: {EPOCHS}")
    print(f"  - Early stopping: Enabled (patience=3)")
    print(f"  - Learning rate reduction: Enabled")
    print(f"  - Device: CPU only\n")
    
    # Force CPU execution during training
    with tf.device('/CPU:0'):
        history_lstm = lstm_model.fit(
            X_train_padded, y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(X_val_padded, y_val),
            callbacks=[early_stopping, model_checkpoint, reduce_lr],
            verbose=1
        )
    
    # Plot training history for LSTM
    plot_training_history(history_lstm)
    
    # Evaluate LSTM model
    lstm_model.load_weights('best_lstm_model.h5')
    evaluate_model(lstm_model, X_test_padded, y_test, 'LSTM')
    
    # Build and train GRU model
    print("\n" + "="*60)
    print("STEP 6: Building and Training GRU Model")
    print("="*60)
    
    gru_model = build_gru_model(vocab_size)
    print("\nModel Architecture:")
    gru_model.summary()
    
    # Callbacks for GRU
    model_checkpoint_gru = ModelCheckpoint(
        'best_gru_model.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # Train GRU model
    print(f"\nStarting training...")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Max epochs: {EPOCHS}")
    print(f"  - Early stopping: Enabled (patience=3)")
    print(f"  - Learning rate reduction: Enabled")
    print(f"  - Device: CPU only\n")
    
    # Force CPU execution during training
    with tf.device('/CPU:0'):
        history_gru = gru_model.fit(
            X_train_padded, y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(X_val_padded, y_val),
            callbacks=[early_stopping, model_checkpoint_gru, reduce_lr],
            verbose=1
        )
    
    # Plot training history for GRU
    plot_training_history(history_gru)
    
    # Evaluate GRU model
    gru_model.load_weights('best_gru_model.h5')
    evaluate_model(gru_model, X_test_padded, y_test, 'GRU')
    
    # Save final models
    print("\n" + "="*60)
    print("Saving models...")
    print("="*60)
    lstm_model.save('lstm_model_final.h5')
    print("  ✓ LSTM model saved as 'lstm_model_final.h5'")
    gru_model.save('gru_model_final.h5')
    print("  ✓ GRU model saved as 'gru_model_final.h5'")
    
    print("\n" + "="*60)
    print("🎉 Training Complete! 🎉")
    print("="*60)
    print("\nGenerated files:")
    print("  - best_lstm_model.h5")
    print("  - best_gru_model.h5")
    print("  - lstm_model_final.h5")
    print("  - gru_model_final.h5")
    print("  - tokenizer.pkl")
    print("  - training_history.png")
    print("  - confusion_matrix_lstm.png")
    print("  - confusion_matrix_gru.png")

if __name__ == "__main__":
    main()

