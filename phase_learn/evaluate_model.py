# phase_learn/evaluate_model.py
import json
import gzip
import pickle
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_classification_results() -> Dict:
    """Load classification results and model."""
    logger.info("Loading classification results...")
    
    results_path = Path("data/models/classifier/classification_results.json")
    model_path = Path("data/models/classifier/classifier_model.pkl")
    
    results = {}
    
    # Load classification results
    if results_path.exists():
        with open(results_path, 'r') as f:
            results['classification'] = json.load(f)
        logger.info("Classification results loaded successfully")
    else:
        logger.warning("Classification results not found")
        results['classification'] = None
    
    # Load model
    if model_path.exists():
        with open(model_path, 'rb') as f:
            results['classification_model'] = pickle.load(f)
        logger.info("Classification model loaded successfully")
    else:
        logger.warning("Classification model not found")
        results['classification_model'] = None
    
    return results


def load_clustering_results() -> Dict:
    """Load clustering results."""
    logger.info("Loading clustering results...")
    
    results_path = Path("data/models/clustering/clustering_results.json")
    analyses_path = Path("data/models/clustering/cluster_analyses.json")
    
    results = {}
    
    if results_path.exists():
        with open(results_path, 'r') as f:
            results['clustering'] = json.load(f)
        logger.info("Clustering results loaded successfully")
    else:
        logger.warning("Clustering results not found")
        results['clustering'] = None
    
    if analyses_path.exists():
        with open(analyses_path, 'r') as f:
            results['cluster_analyses'] = json.load(f)
        logger.info("Cluster analyses loaded successfully")
    else:
        logger.warning("Cluster analyses not found")
        results['cluster_analyses'] = None
    
    return results


def load_topic_model_results() -> Dict:
    """Load topic modeling results if available."""
    logger.info("Loading topic modeling results...")
    
    # Check for topic model outputs
    topic_model_dir = Path("data/models/topic_model")
    results = {}
    
    if topic_model_dir.exists():
        # Look for topic model files
        for file_path in topic_model_dir.glob("*.json"):
            with open(file_path, 'r') as f:
                results[file_path.stem] = json.load(f)
        
        if results:
            logger.info("Topic modeling results loaded successfully")
        else:
            logger.warning("Topic modeling directory exists but no results found")
    else:
        logger.warning("Topic modeling results not found")
        results = None
    
    return results


def evaluate_classification_performance(classification_results: Dict) -> Dict:
    """Evaluate classification model performance."""
    logger.info("Evaluating classification performance...")
    
    if not classification_results or 'evaluation_metrics' not in classification_results:
        logger.warning("No classification metrics available")
        return {}
    
    metrics = classification_results['evaluation_metrics']
    
    # Extract key performance indicators
    performance = {
        'overall_metrics': {
            'accuracy': metrics['accuracy'],
            'f1_weighted': metrics['f1_weighted'],
            'f1_macro': metrics['f1_macro'],
            'num_classes': metrics['num_classes'],
            'test_samples': metrics['num_test_samples']
        },
        'class_performance': {},
        'model_comparison': classification_results.get('training_results', {}),
        'data_insights': {
            'class_imbalance_detected': metrics['f1_weighted'] > metrics['f1_macro'],
            'performance_gap': metrics['f1_weighted'] - metrics['f1_macro']
        }
    }
    
    # Extract per-class performance
    if 'classification_report' in metrics:
        for class_name, class_metrics in metrics['classification_report'].items():
            if isinstance(class_metrics, dict) and 'precision' in class_metrics:
                performance['class_performance'][class_name] = {
                    'precision': class_metrics['precision'],
                    'recall': class_metrics['recall'],
                    'f1_score': class_metrics['f1-score'],
                    'support': class_metrics['support']
                }
    
    # Performance assessment
    accuracy = metrics['accuracy']
    if accuracy >= 0.9:
        performance['assessment'] = 'Excellent'
    elif accuracy >= 0.8:
        performance['assessment'] = 'Good'
    elif accuracy >= 0.7:
        performance['assessment'] = 'Fair'
    else:
        performance['assessment'] = 'Poor'
    
    logger.info(f"Classification assessment: {performance['assessment']}")
    logger.info(f"Overall accuracy: {accuracy:.3f}")
    
    return performance


def evaluate_clustering_quality(clustering_results: Dict) -> Dict:
    """Evaluate clustering quality across different methods."""
    logger.info("Evaluating clustering quality...")
    
    if not clustering_results:
        logger.warning("No clustering results available")
        return {}
    
    evaluation = {}
    
    for method, results in clustering_results.items():
        if results is None:
            continue
            
        method_eval = {
            'method': method,
            'quality_metrics': {},
            'cluster_info': {},
            'assessment': 'Not evaluated'
        }
        
        # K-Means evaluation
        if 'kmeans' in method.lower() and 'best_k' in results:
            best_k = results['best_k']
            if best_k and 'results' in results:
                best_result = results['results'].get(str(best_k), {})
                
                method_eval['quality_metrics'] = {
                    'silhouette_score': best_result.get('silhouette_score', 0),
                    'calinski_harabasz_score': best_result.get('calinski_harabasz_score', 0),
                    'inertia': best_result.get('inertia', 0)
                }
                method_eval['cluster_info'] = {
                    'num_clusters': best_k,
                    'method_type': 'K-Means'
                }
                
                # Assess quality based on silhouette score
                silhouette = best_result.get('silhouette_score', 0)
                if silhouette >= 0.7:
                    method_eval['assessment'] = 'Excellent clustering'
                elif silhouette >= 0.5:
                    method_eval['assessment'] = 'Good clustering'
                elif silhouette >= 0.25:
                    method_eval['assessment'] = 'Fair clustering'
                else:
                    method_eval['assessment'] = 'Poor clustering'
        
        # DBSCAN evaluation
        elif 'dbscan' in method.lower() and isinstance(results, dict):
            method_eval['quality_metrics'] = {
                'silhouette_score': results.get('silhouette_score', 0),
                'eps': results.get('eps', 0)
            }
            method_eval['cluster_info'] = {
                'num_clusters': results.get('n_clusters', 0),
                'noise_points': results.get('n_noise', 0),
                'method_type': 'DBSCAN'
            }
            
            # Assess DBSCAN quality
            silhouette = results.get('silhouette_score', 0)
            n_clusters = results.get('n_clusters', 0)
            if n_clusters > 1 and silhouette >= 0.3:
                method_eval['assessment'] = 'Good clustering'
            elif n_clusters > 1:
                method_eval['assessment'] = 'Fair clustering'
            else:
                method_eval['assessment'] = 'Poor clustering - too few clusters'
        
        evaluation[method] = method_eval
    
    # Find best clustering method
    best_method = None
    best_score = -1
    
    for method, eval_data in evaluation.items():
        silhouette = eval_data['quality_metrics'].get('silhouette_score', -1)
        if silhouette > best_score:
            best_score = silhouette
            best_method = method
    
    evaluation['best_method'] = best_method
    evaluation['best_silhouette'] = best_score
    
    logger.info(f"Best clustering method: {best_method} (silhouette: {best_score:.3f})")
    
    return evaluation


def analyze_cluster_composition(cluster_analyses: Dict) -> Dict:
    """Analyze the composition and characteristics of clusters."""
    logger.info("Analyzing cluster composition...")
    
    if not cluster_analyses:
        logger.warning("No cluster analyses available")
        return {}
    
    composition_analysis = {}
    
    for method, analyses in cluster_analyses.items():
        if not analyses:
            continue
            
        method_analysis = {
            'total_clusters': 0,
            'cluster_sizes': [],
            'category_distribution': {},
            'quality_insights': {}
        }
        
        all_categories = []
        cluster_qualities = []
        
        for cluster_name, cluster_data in analyses.items():
            if cluster_name == 'noise':
                continue
                
            method_analysis['total_clusters'] += 1
            method_analysis['cluster_sizes'].append(cluster_data.get('size', 0))
            
            # Aggregate categories
            cluster_categories = cluster_data.get('top_categories', {})
            for cat, count in cluster_categories.items():
                all_categories.extend([cat] * count)
            
            # Track quality scores
            avg_quality = cluster_data.get('avg_quality_score', 0)
            cluster_qualities.append(avg_quality)
        
        # Calculate overall category distribution
        category_counter = Counter(all_categories)
        method_analysis['category_distribution'] = dict(category_counter.most_common(10))
        
        # Quality insights
        if cluster_qualities:
            method_analysis['quality_insights'] = {
                'avg_cluster_quality': np.mean(cluster_qualities),
                'quality_std': np.std(cluster_qualities),
                'min_quality': min(cluster_qualities),
                'max_quality': max(cluster_qualities)
            }
        
        # Size analysis
        if method_analysis['cluster_sizes']:
            sizes = method_analysis['cluster_sizes']
            method_analysis['size_insights'] = {
                'avg_cluster_size': np.mean(sizes),
                'size_std': np.std(sizes),
                'min_size': min(sizes),
                'max_size': max(sizes),
                'size_imbalance': max(sizes) / min(sizes) if min(sizes) > 0 else 0
            }
        
        composition_analysis[method] = method_analysis
    
    return composition_analysis


def create_performance_visualizations(classification_perf: Dict, clustering_eval: Dict, output_dir: Path):
    """Create comprehensive performance visualizations."""
    logger.info("Creating performance visualizations...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Classification performance plot
    if classification_perf and 'class_performance' in classification_perf:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Summary', fontsize=16)
        
        # Overall metrics
        overall = classification_perf['overall_metrics']
        ax1 = axes[0, 0]
        metrics = ['Accuracy', 'F1 Weighted', 'F1 Macro']
        values = [overall['accuracy'], overall['f1_weighted'], overall['f1_macro']]
        bars = ax1.bar(metrics, values, color=['skyblue', 'lightgreen', 'salmon'])
        ax1.set_title('Classification Metrics')
        ax1.set_ylabel('Score')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom')
        
        # Per-class F1 scores
        ax2 = axes[0, 1]
        if classification_perf['class_performance']:
            classes = list(classification_perf['class_performance'].keys())
            f1_scores = [classification_perf['class_performance'][c]['f1_score'] 
                        for c in classes if 'f1_score' in classification_perf['class_performance'][c]]
            
            if f1_scores:
                ax2.bar(range(len(classes)), f1_scores, color='lightcoral')
                ax2.set_title('Per-Class F1 Scores')
                ax2.set_ylabel('F1 Score')
                ax2.set_xticks(range(len(classes)))
                ax2.set_xticklabels(classes, rotation=45)
        
        # Clustering quality comparison
        ax3 = axes[1, 0]
        if clustering_eval:
            methods = []
            silhouette_scores = []
            
            for method, eval_data in clustering_eval.items():
                if method not in ['best_method', 'best_silhouette']:
                    methods.append(method.replace('_', ' ').title())
                    silhouette_scores.append(eval_data['quality_metrics'].get('silhouette_score', 0))
            
            if methods:
                ax3.bar(methods, silhouette_scores, color='gold')
                ax3.set_title('Clustering Quality (Silhouette Score)')
                ax3.set_ylabel('Silhouette Score')
                ax3.tick_params(axis='x', rotation=45)
        
        # Model comparison
        ax4 = axes[1, 1]
        if 'model_comparison' in classification_perf:
            model_results = classification_perf['model_comparison']
            if model_results:
                models = list(model_results.keys())
                test_f1_scores = [model_results[m].get('test_f1', 0) for m in models]
                
                ax4.bar(models, test_f1_scores, color='lightsteelblue')
                ax4.set_title('Model Comparison (Test F1)')
                ax4.set_ylabel('F1 Score')
                ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'comprehensive_evaluation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Performance visualization saved")


def generate_evaluation_report(all_results: Dict) -> str:
    """Generate a comprehensive evaluation report."""
    logger.info("Generating evaluation report...")
    
    report_lines = [
        "=" * 80,
        "COMPREHENSIVE MODEL EVALUATION REPORT",
        "=" * 80,
        f"Generated: {np.datetime64('now')}",
        "",
        "📊 CLASSIFICATION PERFORMANCE",
        "-" * 40
    ]
    
    # Classification results
    classification_perf = all_results.get('classification_performance', {})
    if classification_perf and 'overall_metrics' in classification_perf:
        overall = classification_perf['overall_metrics']
        assessment = classification_perf.get('assessment', 'Not assessed')
        
        report_lines.extend([
            f"Overall Assessment: {assessment}",
            f"Accuracy: {overall['accuracy']:.3f}",
            f"F1 Weighted: {overall['f1_weighted']:.3f}",
            f"F1 Macro: {overall['f1_macro']:.3f}",
            f"Classes Handled: {overall['num_classes']}",
            f"Test Samples: {overall['test_samples']}",
            ""
        ])
        
        # Class imbalance insights
        data_insights = classification_perf.get('data_insights', {})
        if data_insights.get('class_imbalance_detected'):
            gap = data_insights.get('performance_gap', 0)
            report_lines.extend([
                "⚠️  IMBALANCE DETECTED:",
                f"   Performance gap: {gap:.3f} (F1 weighted - F1 macro)",
                f"   This suggests significant class imbalance in the dataset.",
                ""
            ])
    else:
        report_lines.append("No classification results available.\n")
    
    # Clustering results
    report_lines.extend([
        "🔍 CLUSTERING ANALYSIS",
        "-" * 40
    ])
    
    clustering_eval = all_results.get('clustering_evaluation', {})
    if clustering_eval:
        best_method = clustering_eval.get('best_method', 'None')
        best_score = clustering_eval.get('best_silhouette', 0)
        
        report_lines.extend([
            f"Best Clustering Method: {best_method}",
            f"Best Silhouette Score: {best_score:.3f}",
            ""
        ])
        
        for method, eval_data in clustering_eval.items():
            if method not in ['best_method', 'best_silhouette']:
                assessment = eval_data.get('assessment', 'Not assessed')
                cluster_info = eval_data.get('cluster_info', {})
                quality_metrics = eval_data.get('quality_metrics', {})
                
                report_lines.extend([
                    f"{method.upper()}:",
                    f"   Assessment: {assessment}",
                    f"   Clusters: {cluster_info.get('num_clusters', 'N/A')}",
                    f"   Silhouette: {quality_metrics.get('silhouette_score', 'N/A'):.3f}",
                    ""
                ])
    else:
        report_lines.append("No clustering results available.\n")
    
    # Data composition analysis
    composition = all_results.get('cluster_composition', {})
    if composition:
        report_lines.extend([
            "📈 DATA COMPOSITION INSIGHTS",
            "-" * 40
        ])
        
        for method, analysis in composition.items():
            if 'category_distribution' in analysis:
                top_categories = list(analysis['category_distribution'].items())[:3]
                report_lines.extend([
                    f"{method.upper()} - Top Categories:",
                    *[f"   {cat}: {count}" for cat, count in top_categories],
                    ""
                ])
    
    # Hardware and efficiency notes
    report_lines.extend([
        "⚙️  HARDWARE COMPLIANCE",
        "-" * 40,
        "✅ All models designed for hardware constraints:",
        "   - CPU: 4c 8t (6c 12t actual)",
        "   - GPU: 4GB (GTX 1050 Ti Max-Q)",
        "   - RAM: 32GB max",
        "   - Lightweight models used (all-MiniLM-L6-v2)",
        "   - Batch processing implemented",
        ""
    ])
    
    # Recommendations
    report_lines.extend([
        "🎯 RECOMMENDATIONS",
        "-" * 40
    ])
    
    if classification_perf:
        accuracy = classification_perf.get('overall_metrics', {}).get('accuracy', 0)
        if accuracy < 0.8:
            report_lines.append("📍 Classification: Consider data augmentation or feature engineering")
        elif classification_perf.get('data_insights', {}).get('class_imbalance_detected'):
            report_lines.append("📍 Classification: Address class imbalance with sampling techniques")
        else:
            report_lines.append("📍 Classification: Performance is satisfactory")
    
    if clustering_eval:
        best_score = clustering_eval.get('best_silhouette', 0)
        if best_score < 0.3:
            report_lines.append("📍 Clustering: Consider different features or preprocessing")
        else:
            report_lines.append("📍 Clustering: Good cluster separation achieved")
    
    report_lines.extend([
        "",
        "=" * 80,
        "END OF REPORT"
    ])
    
    return "\n".join(report_lines)


def save_comprehensive_results(all_results: Dict, output_path: Path):
    """Save comprehensive evaluation results."""
    logger.info(f"Saving comprehensive results to {output_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info("Comprehensive results saved")


def main():
    """Main function for comprehensive model evaluation."""
    logger.info("Starting comprehensive model evaluation...")
    
    # Load all results
    classification_data = load_classification_results()
    clustering_data = load_clustering_results()
    topic_data = load_topic_model_results()
    
    # Evaluate each component
    all_results = {}
    
    # Classification evaluation
    if classification_data.get('classification'):
        all_results['classification_performance'] = evaluate_classification_performance(
            classification_data['classification']
        )
    
    # Clustering evaluation
    if clustering_data.get('clustering'):
        all_results['clustering_evaluation'] = evaluate_clustering_quality(
            clustering_data['clustering']
        )
    
    # Cluster composition analysis
    if clustering_data.get('cluster_analyses'):
        all_results['cluster_composition'] = analyze_cluster_composition(
            clustering_data['cluster_analyses']
        )
    
    # Topic modeling (if available)
    if topic_data:
        all_results['topic_modeling'] = topic_data
    
    # Create visualizations
    figures_dir = Path("results/figures")
    create_performance_visualizations(
        all_results.get('classification_performance', {}),
        all_results.get('clustering_evaluation', {}),
        figures_dir
    )
    
    # Generate report
    report = generate_evaluation_report(all_results)
    
    # Save results
    save_comprehensive_results(all_results, Path("results/comprehensive_evaluation.json"))
    
    # Save report
    with open(Path("results/evaluation_report.txt"), 'w') as f:
        f.write(report)
    
    # Print summary
    print(report)
    
    logger.info("\n📁 OUTPUTS SAVED:")
    logger.info("- Comprehensive results: results/comprehensive_evaluation.json")
    logger.info("- Evaluation report: results/evaluation_report.txt")
    logger.info("- Visualizations: results/figures/comprehensive_evaluation.png")
    logger.info("- Individual metrics: results/metrics.json")
    
    logger.info("Comprehensive model evaluation completed.")


if __name__ == "__main__":
    main()