"""
Train Traditional Machine Learning Models for Fake News Detection
1. Logistic Regression with TF-IDF
2. Logistic Regression with Count Vectorizer (Bag of Words)
3. Logistic Regression with combined TF-IDF + Count Vectorizer features
"""

import pandas as pd
import numpy as np
import pickle
import re
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

# Configuration - Anti-overfitting settings
TEST_SPLIT = 0.2
VALIDATION_SPLIT = 0.15
MAX_FEATURES = 500  # Further reduced features to prevent overfitting
NGRAM_RANGE = (1, 1)  # Only unigrams (no bigrams to reduce complexity)
USE_DATA_SUBSET = True  # Use subset of data to prevent memorization
DATA_SUBSET_RATIO = 0.3  # Use only 30% of training data (more aggressive)
REGULARIZATION_C = 0.01  # Very strong regularization (lower C = stronger regularization)
MIN_DF = 5  # Minimum document frequency (ignore rare words - more aggressive)
MAX_DF = 0.9  # Maximum document frequency (ignore very common words - more aggressive)

def preprocess_text(text):
    """Clean and preprocess text data"""
    if pd.isna(text) or text is None:
        return ""
    
    text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def combine_title_text(row):
    """Combine title and text columns"""
    title = preprocess_text(row.get('title', ''))
    text = preprocess_text(row.get('text', ''))
    
    parts = [part for part in [title, text] if part and len(part.strip()) > 0]
    combined = ' '.join(parts) if parts else ''
    
    return combined.strip()

def load_and_prepare_data():
    """Load and prepare data"""
    print("="*60)
    print("Loading and Preparing Data")
    print("="*60)
    
    # Load fake news
    print("Loading Fake.csv...")
    fake_df = pd.read_csv('Fake.csv')
    fake_df['label'] = 0
    
    # Load true news
    print("Loading True.csv...")
    true_df = pd.read_csv('True.csv')
    true_df['label'] = 1
    
    # Combine
    df = pd.concat([fake_df, true_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nTotal samples: {len(df)}")
    print(f"  - Fake news: {len(df[df['label'] == 0])}")
    print(f"  - Real news: {len(df[df['label'] == 1])}")
    
    # Preprocess
    print("\nPreprocessing text...")
    df['combined_text'] = df.apply(combine_title_text, axis=1)
    df = df[df['combined_text'].str.len() > 10]
    
    texts = df['combined_text'].values
    labels = df['label'].values
    
    # Additional shuffle to prevent any ordering bias
    indices = np.random.permutation(len(texts))
    texts = texts[indices]
    labels = labels[indices]
    
    print(f"Final dataset: {len(texts)} samples")
    print(f"  - Fake news: {np.sum(labels == 0):,}")
    print(f"  - Real news: {np.sum(labels == 1):,}")
    
    return texts, labels

def train_logistic_regression_tfidf(X_train, X_val, X_test, y_train, y_val, y_test):
    """Train Logistic Regression with TF-IDF features"""
    print("\n" + "="*60)
    print("MODEL 1: Logistic Regression with TF-IDF")
    print("="*60)
    
    # Create pipeline with strong regularization
    print("Creating TF-IDF vectorizer and Logistic Regression pipeline...")
    print(f"  - Max features: {MAX_FEATURES}")
    print(f"  - N-gram range: {NGRAM_RANGE}")
    print(f"  - Regularization C: {REGULARIZATION_C} (lower = stronger)")
    print(f"  - Min document frequency: {MIN_DF}")
    print(f"  - Max document frequency: {MAX_DF}")
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=MAX_FEATURES,
            ngram_range=NGRAM_RANGE,
            stop_words='english',
            lowercase=True,
            min_df=MIN_DF,  # Ignore words that appear in fewer than MIN_DF documents
            max_df=MAX_DF,  # Ignore words that appear in more than MAX_DF documents
            sublinear_tf=True  # Apply sublinear tf scaling
        )),
        ('lr', LogisticRegression(
            max_iter=1000,
            random_state=42,
            C=REGULARIZATION_C,  # Strong regularization (lower C = stronger)
            penalty='l2',
            solver='lbfgs',
            class_weight='balanced'  # Handle class imbalance
        ))
    ])
    
    # Use subset of training data if enabled
    if USE_DATA_SUBSET:
        subset_size = int(len(X_train) * DATA_SUBSET_RATIO)
        indices = np.random.choice(len(X_train), subset_size, replace=False)
        X_train_subset = [X_train[i] for i in indices]
        y_train_subset = y_train[indices]
        print(f"Training on {len(X_train_subset):,} samples ({DATA_SUBSET_RATIO*100:.0f}% of training data)...")
        pipeline.fit(X_train_subset, y_train_subset)
    else:
        X_train_subset = X_train
        y_train_subset = y_train
        print(f"Training on {len(X_train):,} samples...")
        pipeline.fit(X_train, y_train)
    
    # Cross-validation to check for overfitting
    print("\nPerforming 5-fold cross-validation on training set...")
    if USE_DATA_SUBSET:
        cv_scores = cross_val_score(pipeline, X_train_subset, y_train_subset, cv=5, scoring='f1')
    else:
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1')
    print(f"  CV F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    y_val_pred = pipeline.predict(X_val)
    y_val_proba = pipeline.predict_proba(X_val)[:, 1]
    
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred)
    val_recall = recall_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    val_auc = roc_auc_score(y_val, y_val_proba)
    
    print(f"Validation Accuracy:  {val_accuracy:.4f}")
    print(f"Validation Precision: {val_precision:.4f}")
    print(f"Validation Recall:    {val_recall:.4f}")
    print(f"Validation F1-Score:  {val_f1:.4f}")
    print(f"Validation AUC:       {val_auc:.4f}")
    
    # Check for overfitting (compare train vs validation)
    if USE_DATA_SUBSET:
        y_train_pred = pipeline.predict(X_train_subset)
        train_accuracy = accuracy_score(y_train_subset, y_train_pred)
    else:
        y_train_pred = pipeline.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
    
    overfit_gap = train_accuracy - val_accuracy
    if overfit_gap > 0.1:
        print(f"\n⚠️  WARNING: Possible overfitting detected!")
        print(f"   Train accuracy: {train_accuracy:.4f}")
        print(f"   Validation accuracy: {val_accuracy:.4f}")
        print(f"   Gap: {overfit_gap:.4f} (>0.1 indicates overfitting)")
    else:
        print(f"\n✓ No significant overfitting (train-val gap: {overfit_gap:.4f})")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    y_test_pred = pipeline.predict(X_test)
    y_test_proba = pipeline.predict_proba(X_test)[:, 1]
    
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)
    
    print(f"Test Accuracy:  {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall:    {test_recall:.4f}")
    print(f"Test F1-Score:  {test_f1:.4f}")
    print(f"Test AUC:       {test_auc:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    plot_confusion_matrix(cm, 'Logistic Regression + TF-IDF')
    
    # Save model
    with open('logistic_tfidf_model.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    print("\n✓ Model saved as 'logistic_tfidf_model.pkl'")
    
    return pipeline, {
        'accuracy': test_accuracy,
        'precision': test_precision,
        'recall': test_recall,
        'f1': test_f1,
        'auc': test_auc
    }

def train_logistic_regression_count(X_train, X_val, X_test, y_train, y_val, y_test):
    """Train Logistic Regression with Count Vectorizer (Bag of Words)"""
    print("\n" + "="*60)
    print("MODEL 2: Logistic Regression with Count Vectorizer")
    print("="*60)
    
    # Create pipeline with strong regularization
    print("Creating Count Vectorizer and Logistic Regression pipeline...")
    print(f"  - Max features: {MAX_FEATURES}")
    print(f"  - N-gram range: {NGRAM_RANGE}")
    print(f"  - Regularization C: {REGULARIZATION_C} (lower = stronger)")
    print(f"  - Min document frequency: {MIN_DF}")
    print(f"  - Max document frequency: {MAX_DF}")
    
    pipeline = Pipeline([
        ('count', CountVectorizer(
            max_features=MAX_FEATURES,
            ngram_range=NGRAM_RANGE,
            stop_words='english',
            lowercase=True,
            min_df=MIN_DF,
            max_df=MAX_DF
        )),
        ('lr', LogisticRegression(
            max_iter=1000,
            random_state=42,
            C=REGULARIZATION_C,  # Strong regularization
            penalty='l2',
            solver='lbfgs',
            class_weight='balanced'
        ))
    ])
    
    # Use subset of training data if enabled
    if USE_DATA_SUBSET:
        subset_size = int(len(X_train) * DATA_SUBSET_RATIO)
        indices = np.random.choice(len(X_train), subset_size, replace=False)
        X_train_subset = [X_train[i] for i in indices]
        y_train_subset = y_train[indices]
        print(f"Training on {len(X_train_subset):,} samples ({DATA_SUBSET_RATIO*100:.0f}% of training data)...")
        pipeline.fit(X_train_subset, y_train_subset)
    else:
        print(f"Training on {len(X_train):,} samples...")
        pipeline.fit(X_train, y_train)
    
    # Cross-validation to check for overfitting
    print("\nPerforming 5-fold cross-validation on training set...")
    if USE_DATA_SUBSET:
        cv_scores = cross_val_score(pipeline, X_train_subset, y_train_subset, cv=5, scoring='f1')
    else:
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1')
    print(f"  CV F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    y_val_pred = pipeline.predict(X_val)
    y_val_proba = pipeline.predict_proba(X_val)[:, 1]
    
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred)
    val_recall = recall_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    val_auc = roc_auc_score(y_val, y_val_proba)
    
    print(f"Validation Accuracy:  {val_accuracy:.4f}")
    print(f"Validation Precision: {val_precision:.4f}")
    print(f"Validation Recall:    {val_recall:.4f}")
    print(f"Validation F1-Score:  {val_f1:.4f}")
    print(f"Validation AUC:       {val_auc:.4f}")
    
    # Check for overfitting (compare train vs validation)
    if USE_DATA_SUBSET:
        y_train_pred = pipeline.predict(X_train_subset)
        train_accuracy = accuracy_score(y_train_subset, y_train_pred)
    else:
        y_train_pred = pipeline.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
    
    overfit_gap = train_accuracy - val_accuracy
    if overfit_gap > 0.1:
        print(f"\n⚠️  WARNING: Possible overfitting detected!")
        print(f"   Train accuracy: {train_accuracy:.4f}")
        print(f"   Validation accuracy: {val_accuracy:.4f}")
        print(f"   Gap: {overfit_gap:.4f} (>0.1 indicates overfitting)")
    else:
        print(f"\n✓ No significant overfitting (train-val gap: {overfit_gap:.4f})")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    y_test_pred = pipeline.predict(X_test)
    y_test_proba = pipeline.predict_proba(X_test)[:, 1]
    
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)
    
    print(f"Test Accuracy:  {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall:    {test_recall:.4f}")
    print(f"Test F1-Score:  {test_f1:.4f}")
    print(f"Test AUC:       {test_auc:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    plot_confusion_matrix(cm, 'Logistic Regression + Count Vectorizer')
    
    # Save model
    with open('logistic_count_model.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    print("\n✓ Model saved as 'logistic_count_model.pkl'")
    
    return pipeline, {
        'accuracy': test_accuracy,
        'precision': test_precision,
        'recall': test_recall,
        'f1': test_f1,
        'auc': test_auc
    }

def train_logistic_regression_combined(X_train, X_val, X_test, y_train, y_val, y_test):
    """Train Logistic Regression with combined TF-IDF + Count Vectorizer features"""
    print("\n" + "="*60)
    print("MODEL 3: Logistic Regression with Combined Features (TF-IDF + Count)")
    print("="*60)
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.compose import ColumnTransformer
    
    print("Creating combined feature extractors...")
    print(f"  - Max features per vectorizer: {MAX_FEATURES}")
    print(f"  - N-gram range: {NGRAM_RANGE}")
    print(f"  - Regularization C: {REGULARIZATION_C} (lower = stronger)")
    print(f"  - Min document frequency: {MIN_DF}")
    print(f"  - Max document frequency: {MAX_DF}")
    
    # Create both vectorizers with anti-overfitting settings
    tfidf_vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        stop_words='english',
        lowercase=True,
        min_df=MIN_DF,
        max_df=MAX_DF,
        sublinear_tf=True
    )
    
    count_vectorizer = CountVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        stop_words='english',
        lowercase=True,
        min_df=MIN_DF,
        max_df=MAX_DF
    )
    
    # Use subset of training data if enabled
    if USE_DATA_SUBSET:
        subset_size = int(len(X_train) * DATA_SUBSET_RATIO)
        indices = np.random.choice(len(X_train), subset_size, replace=False)
        X_train_subset = [X_train[i] for i in indices]
        y_train_subset = y_train[indices]
        print(f"\nUsing {len(X_train_subset):,} samples ({DATA_SUBSET_RATIO*100:.0f}% of training data)...")
    else:
        X_train_subset = X_train
        y_train_subset = y_train
    
    # Fit vectorizers
    print("Fitting vectorizers on training data...")
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_subset)
    X_train_count = count_vectorizer.fit_transform(X_train_subset)
    
    # Combine features
    from scipy.sparse import hstack
    X_train_combined = hstack([X_train_tfidf, X_train_count])
    
    print(f"Combined feature shape: {X_train_combined.shape}")
    print(f"  - TF-IDF features: {X_train_tfidf.shape[1]}")
    print(f"  - Count features: {X_train_count.shape[1]}")
    print(f"  - Total features: {X_train_combined.shape[1]}")
    
    # Transform validation and test
    X_val_tfidf = tfidf_vectorizer.transform(X_val)
    X_val_count = count_vectorizer.transform(X_val)
    X_val_combined = hstack([X_val_tfidf, X_val_count])
    
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    X_test_count = count_vectorizer.transform(X_test)
    X_test_combined = hstack([X_test_tfidf, X_test_count])
    
    # Train Logistic Regression with strong regularization
    print("\nTraining Logistic Regression on combined features...")
    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        C=REGULARIZATION_C,  # Strong regularization
        penalty='l2',
        solver='lbfgs',
        class_weight='balanced'
    )
    
    lr_model.fit(X_train_combined, y_train_subset)
    
    # Cross-validation to check for overfitting
    print("\nPerforming 5-fold cross-validation on training set...")
    cv_scores = cross_val_score(lr_model, X_train_combined, y_train_subset, cv=5, scoring='f1')
    print(f"  CV F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    y_val_pred = lr_model.predict(X_val_combined)
    y_val_proba = lr_model.predict_proba(X_val_combined)[:, 1]
    
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred)
    val_recall = recall_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    val_auc = roc_auc_score(y_val, y_val_proba)
    
    print(f"Validation Accuracy:  {val_accuracy:.4f}")
    print(f"Validation Precision: {val_precision:.4f}")
    print(f"Validation Recall:    {val_recall:.4f}")
    print(f"Validation F1-Score:  {val_f1:.4f}")
    print(f"Validation AUC:       {val_auc:.4f}")
    
    # Check for overfitting (compare train vs validation)
    y_train_pred = lr_model.predict(X_train_combined)
    train_accuracy = accuracy_score(y_train_subset, y_train_pred)
    
    overfit_gap = train_accuracy - val_accuracy
    if overfit_gap > 0.1:
        print(f"\n⚠️  WARNING: Possible overfitting detected!")
        print(f"   Train accuracy: {train_accuracy:.4f}")
        print(f"   Validation accuracy: {val_accuracy:.4f}")
        print(f"   Gap: {overfit_gap:.4f} (>0.1 indicates overfitting)")
    else:
        print(f"\n✓ No significant overfitting (train-val gap: {overfit_gap:.4f})")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    y_test_pred = lr_model.predict(X_test_combined)
    y_test_proba = lr_model.predict_proba(X_test_combined)[:, 1]
    
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)
    
    print(f"Test Accuracy:  {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall:    {test_recall:.4f}")
    print(f"Test F1-Score:  {test_f1:.4f}")
    print(f"Test AUC:       {test_auc:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    plot_confusion_matrix(cm, 'Logistic Regression + Combined Features')
    
    # Save model and vectorizers
    model_data = {
        'model': lr_model,
        'tfidf_vectorizer': tfidf_vectorizer,
        'count_vectorizer': count_vectorizer
    }
    with open('logistic_combined_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    print("\n✓ Model saved as 'logistic_combined_model.pkl'")
    
    return model_data, {
        'accuracy': test_accuracy,
        'precision': test_precision,
        'recall': test_recall,
        'f1': test_f1,
        'auc': test_auc
    }

def plot_confusion_matrix(cm, model_name):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Fake', 'Real'], 
                yticklabels=['Fake', 'Real'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    filename = f'confusion_matrix_{model_name.lower().replace(" ", "_").replace("+", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved as '{filename}'")
    plt.close()

def compare_models(results):
    """Compare all three models"""
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    models = ['LR + TF-IDF', 'LR + Count', 'LR + Combined']
    
    comparison_df = pd.DataFrame({
        'Model': models,
        'Accuracy': [results['tfidf']['accuracy'], results['count']['accuracy'], results['combined']['accuracy']],
        'Precision': [results['tfidf']['precision'], results['count']['precision'], results['combined']['precision']],
        'Recall': [results['tfidf']['recall'], results['count']['recall'], results['combined']['recall']],
        'F1-Score': [results['tfidf']['f1'], results['count']['f1'], results['combined']['f1']],
        'AUC': [results['tfidf']['auc'], results['count']['auc'], results['combined']['auc']]
    })
    
    print("\nTest Set Performance Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Find best model
    best_model_idx = comparison_df['F1-Score'].idxmax()
    best_model = comparison_df.loc[best_model_idx, 'Model']
    print(f"\n🏆 Best Model (by F1-Score): {best_model}")
    print(f"   Accuracy: {comparison_df.loc[best_model_idx, 'Accuracy']:.4f}")
    print(f"   F1-Score: {comparison_df.loc[best_model_idx, 'F1-Score']:.4f}")
    print(f"   AUC:      {comparison_df.loc[best_model_idx, 'AUC']:.4f}")
    
    # Save comparison
    comparison_df.to_csv('model_comparison.csv', index=False)
    print("\n✓ Comparison saved as 'model_comparison.csv'")
    
    return comparison_df

def main():
    """Main function"""
    print("="*60)
    print("Traditional ML Models for Fake News Detection")
    print("="*60)
    
    # Load and prepare data
    texts, labels = load_and_prepare_data()
    
    # Split data: 65% train, 15% validation, 20% test
    print("\n" + "="*60)
    print("Splitting data (65/15/20)...")
    print("="*60)
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, labels, test_size=TEST_SPLIT, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VALIDATION_SPLIT/(1-TEST_SPLIT), random_state=42, stratify=y_temp
    )
    
    print(f"Training samples:   {len(X_train):,} (65%)")
    print(f"Validation samples: {len(X_val):,} (15%)")
    print(f"Test samples:       {len(X_test):,} (20%)")
    
    # Train models
    results = {}
    
    # Model 1: Logistic Regression + TF-IDF
    model1, results['tfidf'] = train_logistic_regression_tfidf(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    # Model 2: Logistic Regression + Count Vectorizer
    model2, results['count'] = train_logistic_regression_count(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    # Model 3: Logistic Regression + Combined Features
    model3, results['combined'] = train_logistic_regression_combined(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    # Compare models
    comparison_df = compare_models(results)
    
    print("\n" + "="*60)
    print("🎉 Training Complete! 🎉")
    print("="*60)
    print("\nGenerated files:")
    print("  - logistic_tfidf_model.pkl")
    print("  - logistic_count_model.pkl")
    print("  - logistic_combined_model.pkl")
    print("  - model_comparison.csv")
    print("  - confusion_matrix_*.png (3 files)")

if __name__ == "__main__":
    main()

