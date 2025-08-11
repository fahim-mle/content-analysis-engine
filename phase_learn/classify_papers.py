# phase_learn/classify_papers.py
import json
import gzip
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import pickle
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hardware-aware batch processing
BATCH_SIZE = 1000  # Process in smaller batches to manage memory
MAX_FEATURES = 10000  # Limit TF-IDF features to control memory usage

# Target categories as specified
TARGET_CATEGORIES = [
    'cs.AI', 'cs.CL', 'cs.CV', 'cs.CR', 'cs.CY', 'cs.ET', 'cs.GT', 
    'cs.IR', 'cs.LG', 'cs.MA', 'eess.IV', 'eess.SP', 'cond-mat.dis-nn',
    'cond-mat.stat-mech', 'physics.soc-ph'
]


def load_ml_dataset(filepath: Path) -> List[Dict]:
    """Load the ML dataset from compressed JSON file."""
    logger.info(f"Loading ML dataset from {filepath}")
    
    try:
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            data = json.load(f)
        
        papers = data.get('papers', [])
        logger.info(f"Loaded {len(papers)} papers from dataset")
        return papers
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return []


def prepare_classification_data(papers: List[Dict]) -> Tuple[List[str], List[str]]:
    """Prepare text and labels for classification."""
    logger.info("Preparing classification data...")
    
    texts = []
    labels = []
    
    for paper in papers:
        # Extract text (use title + first part of text for efficiency)
        title = paper.get('title', '')
        text = paper.get('text', '')
        
        # Combine title and first 1000 chars of text to manage memory
        combined_text = f"{title}. {text[:1000]}"
        
        # Extract primary category (first in the list)
        categories = paper.get('categories', [])
        if categories and categories[0] in TARGET_CATEGORIES:
            texts.append(combined_text)
            labels.append(categories[0])
    
    logger.info(f"Prepared {len(texts)} samples for classification")
    logger.info(f"Categories distribution: {set(labels)}")
    
    return texts, labels


def create_tfidf_vectorizer() -> TfidfVectorizer:
    """Create TF-IDF vectorizer with memory-efficient settings."""
    return TfidfVectorizer(
        max_features=MAX_FEATURES,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        lowercase=True,
        strip_accents='unicode'
    )


def train_classifier(texts: List[str], labels: List[str]) -> Dict:
    """Train classification models and return the best one."""
    logger.info("Training classification models...")
    
    # Check class distribution
    from collections import Counter
    class_counts = Counter(labels)
    logger.info(f"Class distribution: {dict(class_counts)}")
    
    # Filter out classes with too few samples for robust training
    min_samples_per_class = 3
    valid_classes = {cls for cls, count in class_counts.items() if count >= min_samples_per_class}
    
    if len(valid_classes) < 2:
        logger.warning("Insufficient classes for classification. Using all available data.")
        valid_classes = set(labels)
    
    # Filter data to include only valid classes
    filtered_texts = []
    filtered_labels = []
    for text, label in zip(texts, labels):
        if label in valid_classes:
            filtered_texts.append(text)
            filtered_labels.append(label)
    
    logger.info(f"Filtered to {len(filtered_texts)} samples across {len(valid_classes)} classes")
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(filtered_labels)
    
    # Create TF-IDF vectors
    vectorizer = create_tfidf_vectorizer()
    X = vectorizer.fit_transform(filtered_texts)
    
    logger.info(f"TF-IDF matrix shape: {X.shape}")
    
    # Split data - use stratify only if all classes have at least 2 samples
    stratify_param = None
    if all(class_counts[cls] >= 2 for cls in valid_classes):
        stratify_param = encoded_labels
        logger.info("Using stratified split")
    else:
        logger.info("Using random split due to class imbalance")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, encoded_labels, test_size=0.3, random_state=42, 
        stratify=stratify_param
    )
    
    # Train multiple models
    models = {
        'SVM': SVC(kernel='rbf', random_state=42, probability=True),
        'LogisticRegression': LogisticRegression(
            random_state=42, max_iter=1000, 
            class_weight='balanced'
        )
    }
    
    best_model = None
    best_score = 0
    best_name = None
    results = {}
    
    for name, model in models.items():
        logger.info(f"Training {name}...")
        
        # Cross-validation with adaptive CV folds
        cv_folds = min(5, len(y_train) // 2)  # Ensure at least 2 samples per fold
        cv_folds = max(2, cv_folds)  # Minimum 2 folds
        
        if cv_folds >= 2:
            cv_scores = cross_val_score(
                model, X_train, y_train, cv=cv_folds, scoring='f1_weighted'
            )
        else:
            # If CV not possible, use training score as proxy
            model.fit(X_train, y_train)
            cv_scores = np.array([f1_score(y_train, model.predict(X_train), average='weighted')])
            logger.warning(f"CV not possible for {name}, using training score")
        
        # Train on full training set
        model.fit(X_train, y_train)
        
        # Evaluate on test set
        test_score = f1_score(y_test, model.predict(X_test), average='weighted')
        
        results[name] = {
            'cv_score_mean': cv_scores.mean(),
            'cv_score_std': cv_scores.std(),
            'test_f1': test_score
        }
        
        logger.info(f"{name} - CV F1: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        logger.info(f"{name} - Test F1: {test_score:.3f}")
        
        if test_score > best_score:
            best_score = test_score
            best_model = model
            best_name = name
    
    logger.info(f"Best model: {best_name} with F1 score: {best_score:.3f}")
    
    return {
        'model': best_model,
        'vectorizer': vectorizer,
        'label_encoder': label_encoder,
        'X_test': X_test,
        'y_test': y_test,
        'results': results,
        'best_name': best_name
    }


def evaluate_classifier(model, vectorizer, label_encoder, X_test, y_test) -> Dict:
    """Evaluate the trained classifier and return metrics."""
    logger.info("Evaluating classifier...")
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    # Classification report
    target_names = label_encoder.classes_
    report = classification_report(
        y_test, y_pred, 
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
    
    metrics = {
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'classification_report': report,
        'num_classes': len(target_names),
        'num_test_samples': len(y_test)
    }
    
    logger.info(f"Classification Results:")
    logger.info(f"Accuracy: {accuracy:.3f}")
    logger.info(f"F1 (weighted): {f1_weighted:.3f}")
    logger.info(f"F1 (macro): {f1_macro:.3f}")
    
    return metrics


def save_classifier_model(model, vectorizer, label_encoder, filepath: Path) -> bool:
    """Save the trained classifier model and components."""
    try:
        filepath.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': model,
            'vectorizer': vectorizer,
            'label_encoder': label_encoder,
            'target_categories': TARGET_CATEGORIES
        }
        
        model_file = filepath / 'classifier_model.pkl'
        with open(model_file, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {model_file}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        return False


def predict_categories(texts: List[str], model_path: Path) -> List[str]:
    """Load model and predict categories for new texts."""
    try:
        with open(model_path / 'classifier_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        vectorizer = model_data['vectorizer']
        label_encoder = model_data['label_encoder']
        
        # Transform texts
        X = vectorizer.transform(texts)
        
        # Predict
        predictions = model.predict(X)
        
        # Decode labels
        return label_encoder.inverse_transform(predictions).tolist()
        
    except Exception as e:
        logger.error(f"Failed to predict categories: {e}")
        return []


def main():
    """Main function to train and evaluate the paper classifier."""
    logger.info("Starting paper classification training...")
    
    # Paths
    data_path = Path("data/metadata/ml_dataset.json.gz")
    model_path = Path("data/models/classifier")
    
    # Load dataset
    papers = load_ml_dataset(data_path)
    if not papers:
        logger.error("No papers loaded. Exiting.")
        return
    
    # Prepare data
    texts, labels = prepare_classification_data(papers)
    if len(texts) < 10:
        logger.error("Insufficient data for classification. Exiting.")
        return
    
    # Train classifier
    training_results = train_classifier(texts, labels)
    
    # Evaluate classifier
    evaluation_metrics = evaluate_classifier(
        training_results['model'],
        training_results['vectorizer'], 
        training_results['label_encoder'],
        training_results['X_test'],
        training_results['y_test']
    )
    
    # Save model
    save_success = save_classifier_model(
        training_results['model'],
        training_results['vectorizer'],
        training_results['label_encoder'],
        model_path
    )
    
    # Save results
    if save_success:
        results_file = model_path / 'classification_results.json'
        final_results = {
            'training_results': training_results['results'],
            'evaluation_metrics': evaluation_metrics,
            'best_model': training_results['best_name']
        }
        
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_file}")
    
    logger.info("Paper classification training completed.")


if __name__ == "__main__":
    main()