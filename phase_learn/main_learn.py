# phase_learn/main_learn.py
"""Unified execution system for Phase 3 (Machine Learning) pipeline."""

import time
from pathlib import Path
from typing import Dict, List, Optional, Any

from .utils import get_logger, load_ml_dataset, save_json_results
from .config import ML_DATASET_FILE, MODELS_DIR, REPORTS_DIR
from .encode_embeddings import process_paper_embeddings
from .train_topic_model import run_topic_modeling
from .classify_papers import main as run_classification
from .cluster_papers import main as run_clustering
from .evaluate_model import main as run_evaluation

logger = get_logger(__name__)

class MLPipelineOrchestrator:
    """Orchestrates the complete ML pipeline execution."""
    
    def __init__(self):
        self.start_time = None
        self.results = {}
        self.pipeline_status = {
            'dataset_loaded': False,
            'embeddings_generated': False,
            'classification_completed': False,
            'clustering_completed': False,
            'evaluation_completed': False
        }
    
    def validate_dataset(self) -> bool:
        """Validate that the ML dataset exists and is accessible."""
        logger.info("Validating ML dataset...")
        
        if not ML_DATASET_FILE.exists():
            logger.error(f"ML dataset not found at: {ML_DATASET_FILE}")
            return False
            
        dataset = load_ml_dataset(ML_DATASET_FILE)
        if not dataset:
            logger.error("Failed to load ML dataset")
            return False
        
        papers = dataset.get('papers', [])
        chunks = dataset.get('chunks', [])
        
        if len(papers) < 10:
            logger.error(f"Insufficient papers in dataset: {len(papers)}")
            return False
            
        logger.info(f"Dataset validated: {len(papers)} papers, {len(chunks)} chunks")
        self.pipeline_status['dataset_loaded'] = True
        return True
    
    def run_embedding_generation(self) -> bool:
        """Execute embedding generation with error handling."""
        logger.info("=== STEP 1: EMBEDDING GENERATION ===")
        
        try:
            process_paper_embeddings()
            self.pipeline_status['embeddings_generated'] = True
            logger.info("Embedding generation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}", exc_info=True)
            return False
    
    def run_topic_modeling(self) -> bool:
        """Execute topic modeling with error handling."""
        logger.info("=== STEP 2: TOPIC MODELING ===")
        
        try:
            run_topic_modeling()
            logger.info("Topic modeling completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Topic modeling failed: {e}", exc_info=True)
            return False
    
    def run_paper_classification(self) -> bool:
        """Execute paper classification with error handling."""
        logger.info("=== STEP 3: PAPER CLASSIFICATION ===")
        
        try:
            run_classification()
            self.pipeline_status['classification_completed'] = True
            logger.info("Paper classification completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Paper classification failed: {e}", exc_info=True)
            return False
    
    def run_paper_clustering(self) -> bool:
        """Execute paper clustering with error handling."""
        logger.info("=== STEP 4: PAPER CLUSTERING ===")
        
        try:
            run_clustering()
            self.pipeline_status['clustering_completed'] = True
            logger.info("Paper clustering completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Paper clustering failed: {e}", exc_info=True)
            return False
    
    def run_comprehensive_evaluation(self) -> bool:
        """Execute comprehensive model evaluation."""
        logger.info("=== STEP 5: COMPREHENSIVE EVALUATION ===")
        
        try:
            run_evaluation()
            self.pipeline_status['evaluation_completed'] = True
            logger.info("Comprehensive evaluation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Comprehensive evaluation failed: {e}", exc_info=True)
            return False
    
    def generate_pipeline_summary(self) -> Dict[str, Any]:
        """Generate comprehensive pipeline execution summary."""
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        # Count completed steps
        completed_steps = sum(1 for status in self.pipeline_status.values() if status)
        total_steps = len(self.pipeline_status)
        
        # Determine overall success
        critical_steps = ['dataset_loaded', 'classification_completed', 'clustering_completed']
        critical_success = all(self.pipeline_status.get(step, False) for step in critical_steps)
        
        summary = {
            'execution_metadata': {
                'total_execution_time_seconds': elapsed_time,
                'execution_time_formatted': f"{elapsed_time//3600:.0f}h {(elapsed_time%3600)//60:.0f}m {elapsed_time%60:.0f}s",
                'completed_steps': completed_steps,
                'total_steps': total_steps,
                'success_rate': completed_steps / total_steps,
                'overall_success': critical_success
            },
            'pipeline_status': self.pipeline_status,
            'step_details': {
                'step_1_embeddings': {
                    'status': 'completed' if self.pipeline_status['embeddings_generated'] else 'failed',
                    'description': 'Generate sentence embeddings using all-MiniLM-L6-v2'
                },
                'step_2_topic_modeling': {
                    'status': 'completed' if self.pipeline_status.get('topic_modeling', False) else 'failed',
                    'description': 'Train topic model using BERTopic'
                },
                'step_3_classification': {
                    'status': 'completed' if self.pipeline_status['classification_completed'] else 'failed',
                    'description': 'Train paper classification models (SVM, Logistic Regression)'
                },
                'step_4_clustering': {
                    'status': 'completed' if self.pipeline_status['clustering_completed'] else 'failed',
                    'description': 'Perform clustering analysis (K-Means, DBSCAN)'
                },
                'step_5_evaluation': {
                    'status': 'completed' if self.pipeline_status['evaluation_completed'] else 'failed',
                    'description': 'Generate comprehensive evaluation and visualizations'
                }
            },
            'outputs_generated': {
                'models_directory': str(MODELS_DIR),
                'reports_directory': str(REPORTS_DIR),
                'expected_files': [
                    'data/models/classifier/classifier_model.pkl',
                    'data/models/classifier/classification_results.json',
                    'data/models/clustering/clustering_results.json',
                    'data/models/clustering/cluster_analyses.json',
                    'results/comprehensive_evaluation.json',
                    'results/evaluation_report.txt',
                    'results/figures/comprehensive_evaluation.png'
                ]
            },
            'hardware_compliance': {
                'cpu_optimized': 'Batch processing for 4c 8t CPU',
                'gpu_optimized': 'Lightweight models for 4GB GPU',
                'memory_optimized': 'Streaming and batching for 32GB RAM',
                'model_size': 'all-MiniLM-L6-v2 (90MB) instead of large models'
            }
        }
        
        return summary
    
    def save_execution_log(self, summary: Dict[str, Any]) -> None:
        """Save detailed execution log."""
        try:
            log_file = REPORTS_DIR / "ml_pipeline_execution.json"
            save_json_results(summary, log_file.name)
            logger.info(f"Execution log saved to: {log_file}")
            
        except Exception as e:
            logger.error(f"Failed to save execution log: {e}")
    
    def print_final_report(self, summary: Dict[str, Any]) -> None:
        """Print comprehensive final report to console."""
        metadata = summary['execution_metadata']
        
        print("\n" + "=" * 80)
        print("PHASE 3 (MACHINE LEARNING) - UNIFIED EXECUTION REPORT")
        print("=" * 80)
        print(f"Total Execution Time: {metadata['execution_time_formatted']}")
        print(f"Steps Completed: {metadata['completed_steps']}/{metadata['total_steps']}")
        print(f"Success Rate: {metadata['success_rate']:.1%}")
        print(f"Overall Success: {'✅ YES' if metadata['overall_success'] else '❌ NO'}")
        print()
        
        print("STEP EXECUTION SUMMARY:")
        print("-" * 40)
        for step_key, step_info in summary['step_details'].items():
            status_icon = "✅" if step_info['status'] == 'completed' else "❌"
            step_num = step_key.split('_')[1]
            print(f"{status_icon} Step {step_num}: {step_info['description']}")
        print()
        
        print("GENERATED OUTPUTS:")
        print("-" * 40)
        for file_path in summary['outputs_generated']['expected_files']:
            if Path(file_path).exists():
                print(f"✅ {file_path}")
            else:
                print(f"❌ {file_path}")
        print()
        
        print("HARDWARE OPTIMIZATION:")
        print("-" * 40)
        for key, value in summary['hardware_compliance'].items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        print()
        
        if metadata['overall_success']:
            print("🎉 PHASE 3 PIPELINE COMPLETED SUCCESSFULLY!")
            print("All ML models have been trained and evaluated.")
            print("Check the results/ directory for comprehensive analysis.")
        else:
            print("⚠️  PHASE 3 PIPELINE COMPLETED WITH ISSUES")
            print("Some steps failed. Check logs for details.")
        
        print("=" * 80)


def run_learn_phase() -> bool:
    """
    Execute the complete unified ML pipeline.
    
    Returns:
        True if pipeline completed successfully, False otherwise
    """
    orchestrator = MLPipelineOrchestrator()
    orchestrator.start_time = time.time()
    
    logger.info("🚀 Starting Phase 3 (Machine Learning) - Unified Pipeline")
    logger.info("Hardware: 4c 8t CPU, 4GB GPU, 32GB RAM - Optimized execution")
    
    # Step 0: Validate dataset
    if not orchestrator.validate_dataset():
        logger.error("Dataset validation failed. Cannot proceed with ML pipeline.")
        return False
    
    # Execute pipeline steps with graceful error handling
    steps_success = []
    
    # Step 1: Embedding Generation
    steps_success.append(orchestrator.run_embedding_generation())
    
    # Step 2: Topic Modeling (optional, continues if fails)
    topic_success = orchestrator.run_topic_modeling()
    if topic_success:
        orchestrator.pipeline_status['topic_modeling'] = True
    
    # Step 3: Classification (critical)
    steps_success.append(orchestrator.run_paper_classification())
    
    # Step 4: Clustering (critical)
    steps_success.append(orchestrator.run_paper_clustering())
    
    # Step 5: Evaluation (runs if any models exist)
    if any(steps_success):
        steps_success.append(orchestrator.run_comprehensive_evaluation())
    
    # Generate comprehensive summary
    summary = orchestrator.generate_pipeline_summary()
    
    # Save execution log
    orchestrator.save_execution_log(summary)
    
    # Print final report
    orchestrator.print_final_report(summary)
    
    # Determine overall success
    critical_success = summary['execution_metadata']['overall_success']
    
    if critical_success:
        logger.info("✅ Phase 3 ML pipeline completed successfully!")
        return True
    else:
        logger.error("❌ Phase 3 ML pipeline completed with critical failures.")
        return False


if __name__ == "__main__":
    success = run_learn_phase()
    exit(0 if success else 1)
