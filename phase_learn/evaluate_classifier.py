# phase_learn/evaluate_classifier.py
import json
import gzip
import pickle
from pathlib import Path
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


def load_classifier(model_path: Path):
    """Load the trained classifier model."""
    try:
        with open(model_path / 'classifier_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except Exception as e:
        logger.error(f"Failed to load classifier: {e}")
        return None


def predict_paper_categories(texts: List[str], model_data: Dict) -> List[Dict]:
    """Predict categories for new papers."""
    model = model_data['model']
    vectorizer = model_data['vectorizer']
    label_encoder = model_data['label_encoder']
    
    # Transform texts
    X = vectorizer.transform(texts)
    
    # Get predictions and probabilities
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    results = []
    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
        # Get class name
        pred_class = label_encoder.inverse_transform([pred])[0]
        
        # Get confidence (max probability)
        confidence = max(prob)
        
        # Get all class probabilities
        class_probs = {}
        for j, class_name in enumerate(label_encoder.classes_):
            class_probs[class_name] = float(prob[j])
        
        results.append({
            'text_index': i,
            'predicted_category': pred_class,
            'confidence': float(confidence),
            'all_probabilities': class_probs
        })
    
    return results


def demo_predictions():
    """Demonstrate predictions on sample texts."""
    logger.info("Running prediction demo...")
    
    model_path = Path("data/models/classifier")
    model_data = load_classifier(model_path)
    
    if not model_data:
        logger.error("Could not load classifier model")
        return
    
    # Sample texts for demonstration
    sample_texts = [
        "Deep learning neural networks for computer vision and image recognition tasks.",
        "Natural language processing and computational linguistics methods for text analysis.",
        "Artificial intelligence algorithms for machine learning and pattern recognition."
    ]
    
    predictions = predict_paper_categories(sample_texts, model_data)
    
    print("\n" + "="*60)
    print("CLASSIFICATION DEMO - SAMPLE PREDICTIONS")
    print("="*60)
    
    for i, (text, pred) in enumerate(zip(sample_texts, predictions)):
        print(f"\nSample {i+1}:")
        print(f"Text: {text}")
        print(f"Predicted Category: {pred['predicted_category']}")
        print(f"Confidence: {pred['confidence']:.3f}")
        print("All Probabilities:")
        for cat, prob in pred['all_probabilities'].items():
            print(f"  {cat}: {prob:.3f}")
    
    print("="*60)


def evaluate_on_test_samples():
    """Evaluate classifier on a few test samples from the dataset."""
    logger.info("Evaluating on test samples...")
    
    # Load dataset
    data_path = Path("data/metadata/ml_dataset.json.gz")
    try:
        with gzip.open(data_path, 'rt', encoding='utf-8') as f:
            data = json.load(f)
        papers = data.get('papers', [])
    except Exception as e:
        logger.error(f"Failed to load test dataset: {e}")
        return
    
    # Load classifier
    model_path = Path("data/models/classifier")
    model_data = load_classifier(model_path)
    
    if not model_data:
        return
    
    # Take a few samples for evaluation
    test_samples = papers[:5]
    texts = []
    true_labels = []
    
    for paper in test_samples:
        title = paper.get('title', '')
        text = paper.get('text', '')
        combined_text = f"{title}. {text[:1000]}"
        texts.append(combined_text)
        
        categories = paper.get('categories', [])
        true_labels.append(categories[0] if categories else 'unknown')
    
    # Get predictions
    predictions = predict_paper_categories(texts, model_data)
    
    print("\n" + "="*80)
    print("CLASSIFIER EVALUATION ON TEST SAMPLES")
    print("="*80)
    
    for i, (paper, pred, true_label) in enumerate(zip(test_samples, predictions, true_labels)):
        print(f"\nTest Sample {i+1}:")
        print(f"Paper ID: {paper.get('arxiv_id', 'unknown')}")
        print(f"True Category: {true_label}")
        print(f"Predicted Category: {pred['predicted_category']}")
        print(f"Confidence: {pred['confidence']:.3f}")
        print(f"Correct: {'✓' if pred['predicted_category'] == true_label else '✗'}")
    
    print("="*80)


def save_final_metrics():
    """Save comprehensive final metrics."""
    logger.info("Saving final evaluation metrics...")
    
    results_path = Path("data/models/classifier/classification_results.json")
    if not results_path.exists():
        logger.error("Classification results not found")
        return
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Create comprehensive metrics summary
    final_metrics = {
        'model_performance': {
            'best_model': results['best_model'],
            'accuracy': results['evaluation_metrics']['accuracy'],
            'f1_weighted': results['evaluation_metrics']['f1_weighted'],
            'f1_macro': results['evaluation_metrics']['f1_macro'],
            'num_classes': results['evaluation_metrics']['num_classes'],
            'test_samples': results['evaluation_metrics']['num_test_samples']
        },
        'class_performance': {},
        'training_info': {
            'models_tested': list(results['training_results'].keys()),
            'best_cv_score': results['training_results'][results['best_model']]['cv_score_mean'],
            'cv_std': results['training_results'][results['best_model']]['cv_score_std']
        },
        'dataset_info': {
            'total_papers': 296,
            'filtered_for_training': 290,
            'major_classes': ['cs.AI', 'cs.CL', 'cs.CV'],
            'class_imbalance': 'severe (cs.AI: 279, cs.CL: 6, cs.CV: 5)'
        }
    }
    
    # Extract class-wise performance
    for class_name, metrics in results['evaluation_metrics']['classification_report'].items():
        if isinstance(metrics, dict) and 'precision' in metrics:
            final_metrics['class_performance'][class_name] = {
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1-score'],
                'support': int(metrics['support'])
            }
    
    # Save final metrics
    output_path = Path("results/metrics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    logger.info(f"Final metrics saved to {output_path}")
    
    return final_metrics


def main():
    """Main evaluation function."""
    print("\n🔍 PAPER CLASSIFIER EVALUATION")
    print("="*50)
    
    # Run demo predictions
    demo_predictions()
    
    # Evaluate on test samples
    evaluate_on_test_samples()
    
    # Save final metrics
    final_metrics = save_final_metrics()
    
    if final_metrics:
        print(f"\n📊 FINAL SUMMARY:")
        print(f"Best Model: {final_metrics['model_performance']['best_model']}")
        print(f"Overall Accuracy: {final_metrics['model_performance']['accuracy']:.3f}")
        print(f"Weighted F1: {final_metrics['model_performance']['f1_weighted']:.3f}")
        print(f"Classes Handled: {final_metrics['model_performance']['num_classes']}")
        print(f"Training Data: {final_metrics['dataset_info']['filtered_for_training']} papers")
        print(f"\n📁 Outputs:")
        print(f"- Model: data/models/classifier/classifier_model.pkl")
        print(f"- Metrics: results/metrics.json")
        print(f"- Figures: results/figures/")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()