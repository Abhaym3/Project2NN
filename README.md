# Fake News Detection Neural Network

A deep learning project that uses LSTM and GRU neural networks to classify fake news vs real news articles.

## Features

- **Dual Model Architecture**: Implements both LSTM and GRU models for comparison
- **Bidirectional Layers**: Uses bidirectional LSTM/GRU for better context understanding
- **Comprehensive Preprocessing**: Text cleaning, tokenization, and sequence padding
- **80/10/10 Data Split**: Training (80%), Validation (10%), Test (10%)
- **Model Evaluation**: Includes accuracy, precision, recall, F1-score, and confusion matrices
- **Visualization**: Training history plots and confusion matrices

## Requirements

- Python 3.7 or higher
- TensorFlow 2.15.0
- Keras 2.15.0
- Other dependencies listed in `requirements.txt`

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. The script will automatically download NLTK data (punkt tokenizer and stopwords) on first run.

## Data Files

Make sure you have the following CSV files in the project directory:
- `Fake.csv` - Contains fake news articles
- `True.csv` - Contains real news articles
- `val.csv` - Optional validation file (not used in current implementation)

## Usage

Run the main script:
```bash
python fake_news_detector.py
```

## Model Architecture

### LSTM Model
- Embedding Layer (128 dimensions)
- Bidirectional LSTM (128 units) with return sequences
- Bidirectional LSTM (64 units)
- Dense layers with dropout
- Sigmoid output for binary classification

### GRU Model
- Embedding Layer (128 dimensions)
- Bidirectional GRU (128 units) with return sequences
- Bidirectional GRU (64 units)
- Dense layers with dropout
- Sigmoid output for binary classification

## Configuration

You can modify these parameters in the script:
- `MAX_SEQUENCE_LENGTH`: Maximum length of input sequences (default: 500)
- `MAX_NB_WORDS`: Maximum vocabulary size (default: 10000)
- `EMBEDDING_DIM`: Embedding dimension (default: 128)
- `LSTM_UNITS`: Number of LSTM units (default: 128)
- `GRU_UNITS`: Number of GRU units (default: 128)
- `BATCH_SIZE`: Training batch size (default: 64)
- `EPOCHS`: Maximum number of training epochs (default: 20)

## Output Files

After training, the following files will be generated:
- `best_lstm_model.h5` - Best LSTM model (based on validation loss)
- `best_gru_model.h5` - Best GRU model (based on validation loss)
- `lstm_model_final.h5` - Final LSTM model
- `gru_model_final.h5` - Final GRU model
- `tokenizer.pkl` - Saved tokenizer for future predictions
- `training_history.png` - Training/validation metrics plots
- `confusion_matrix_lstm.png` - LSTM confusion matrix
- `confusion_matrix_gru.png` - GRU confusion matrix

## Model Performance

The script will display:
- Training progress for each epoch
- Test set metrics (Accuracy, Precision, Recall, F1-Score)
- Classification report
- Confusion matrices

## Training Features

- **Early Stopping**: Stops training if validation loss doesn't improve for 3 epochs
- **Model Checkpointing**: Saves the best model based on validation loss
- **Learning Rate Reduction**: Reduces learning rate when validation loss plateaus
- **Stratified Splitting**: Maintains class balance in train/val/test splits

## Notes

- The script combines title and text columns for better feature extraction
- Text preprocessing includes: lowercase conversion, URL removal, special character removal
- Models use binary crossentropy loss and Adam optimizer
- Both models are trained on the same data for fair comparison

## Troubleshooting

If you encounter memory issues:
- Reduce `MAX_SEQUENCE_LENGTH`
- Reduce `BATCH_SIZE`
- Reduce `LSTM_UNITS` or `GRU_UNITS`

If training is too slow:
- Reduce `MAX_NB_WORDS`
- Reduce `EMBEDDING_DIM`
- Use a smaller subset of data for testing

