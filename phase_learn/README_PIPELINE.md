# Phase 3 (Machine Learning) - Unified Pipeline Execution System

## Overview

This unified execution system orchestrates all Phase 3 ML modules into a cohesive pipeline that handles:
- Dataset validation and loading
- Embedding generation  
- Classification training
- Clustering analysis
- Comprehensive evaluation and reporting

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Dataset        │    │  Embedding      │    │  Classification │
│  Validation     │───▶│  Generation     │───▶│  Training       │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐            │
│  Comprehensive  │    │  Clustering     │◀───────────┘
│  Evaluation     │◀───│  Analysis       │
│                 │    │                 │
└─────────────────┘    └─────────────────┘
```

## Execution Methods

### Method 1: Direct Module Execution
```bash
python -m phase_learn.main_learn
```

### Method 2: Python API
```python
from phase_learn import run_learn_phase

# Execute full pipeline
success = run_learn_phase()
if success:
    print("✅ Pipeline completed successfully!")
else:
    print("❌ Pipeline encountered errors.")
```

### Method 3: Orchestrator Class (Advanced)
```python
from phase_learn.main_learn import MLPipelineOrchestrator

# Initialize orchestrator
orchestrator = MLPipelineOrchestrator()

# Run individual steps
orchestrator.validate_dataset()
orchestrator.run_embedding_generation()
orchestrator.run_paper_classification()
orchestrator.run_paper_clustering()
orchestrator.run_comprehensive_evaluation()

# Generate summary and report
summary = orchestrator.generate_pipeline_summary()
orchestrator.print_final_report(summary)
```

## Pipeline Steps

### Step 1: Embedding Generation
- **Module**: `encode_embeddings.py`
- **Purpose**: Generate sentence embeddings using all-MiniLM-L6-v2
- **Input**: `data/metadata/ml_dataset.json.gz`
- **Output**: Embeddings saved to embedding store
- **Hardware**: Optimized for 4GB GPU with batch processing

### Step 2: Topic Modeling (Optional)
- **Module**: `train_topic_model.py`
- **Purpose**: Discover themes using BERTopic
- **Input**: Generated embeddings
- **Output**: Topic model and analysis
- **Note**: Pipeline continues even if this step fails

### Step 3: Classification Training
- **Module**: `classify_papers.py`
- **Purpose**: Train SVM and Logistic Regression models
- **Input**: Paper texts and categories
- **Output**: `data/models/classifier/` directory with trained models
- **Metrics**: F1-score, accuracy, precision, recall

### Step 4: Clustering Analysis
- **Module**: `cluster_papers.py`
- **Purpose**: Perform K-Means and DBSCAN clustering
- **Input**: Paper embeddings and metadata
- **Output**: `data/models/clustering/` directory with results
- **Visualizations**: PCA plots saved to `results/figures/`

### Step 5: Comprehensive Evaluation
- **Module**: `evaluate_model.py`
- **Purpose**: Generate unified metrics and visualizations
- **Input**: All previous step outputs
- **Output**: `results/` directory with comprehensive analysis
- **Report**: Text and JSON formats

## Expected Outputs

After successful execution, the following files will be created:

### Models Directory (`data/models/`)
```
classifier/
├── classifier_model.pkl         # Trained classification model
└── classification_results.json  # Classification metrics

clustering/
├── clustering_results.json      # Clustering results
└── cluster_analyses.json        # Cluster composition analysis

topic_model/                     # (Optional, if topic modeling succeeds)
└── bertopic_model.pkl          # Trained topic model
```

### Results Directory (`results/`)
```
├── comprehensive_evaluation.json   # Complete evaluation metrics
├── evaluation_report.txt          # Human-readable report  
└── figures/
    └── comprehensive_evaluation.png # Performance visualizations
```

### Reports Directory (`reports/`)
```
└── ml_pipeline_execution.json     # Detailed execution log
```

### Logs Directory (`data/logs/`)
```
└── learn.log                      # Execution logs
```

## Hardware Optimization Features

- **CPU**: Batch processing optimized for 4c 8t processors
- **GPU**: Lightweight models (all-MiniLM-L6-v2, 90MB) for 4GB GPU memory
- **RAM**: Memory-efficient streaming and batching for 32GB systems
- **Models**: Hardware-appropriate model selection avoiding large transformers

## Error Handling

The pipeline includes comprehensive error handling:

1. **Dataset Validation**: Ensures sufficient data before proceeding
2. **Step-by-Step Validation**: Each step validates inputs before execution
3. **Graceful Degradation**: Non-critical steps (topic modeling) won't stop pipeline
4. **Detailed Logging**: All errors logged with full stack traces
5. **Status Tracking**: Real-time pipeline status monitoring

## Monitoring Execution

### Real-Time Logs
```bash
tail -f data/logs/learn.log
```

### Pipeline Status Check
```python
from phase_learn.main_learn import MLPipelineOrchestrator
orchestrator = MLPipelineOrchestrator()
print(orchestrator.pipeline_status)
```

### Quick Test
```bash
python test_phase3_pipeline.py
```

## Integration with Main Pipeline

The Phase 3 execution integrates seamlessly with the main project pipeline:

```python
# main.py
from phase_learn import run_learn_phase

def main():
    # ... Phase 1 and 2 execution ...
    
    # Phase 3: Machine Learning
    print("🤖 Starting Phase 3: Machine Learning")
    ml_success = run_learn_phase()
    
    if ml_success:
        print("✅ Phase 3 completed successfully")
    else:
        print("❌ Phase 3 encountered issues")
        # Pipeline can continue or stop based on requirements
```

## Troubleshooting

### Common Issues

1. **Dataset Not Found**
   - Ensure Phase 2 has completed successfully
   - Check `data/metadata/ml_dataset.json.gz` exists

2. **Memory Issues**
   - Reduce batch sizes in `config.py`
   - Monitor system memory usage

3. **GPU Out of Memory**
   - The pipeline uses CPU fallback automatically
   - Reduce `BATCH_SIZE` in individual modules

4. **Model Training Failures**
   - Check data quality and class distribution
   - Review logs for specific error messages

### Performance Tuning

- Adjust `BATCH_SIZE` in `config.py` based on available RAM
- Modify embedding model in `config.py` if needed
- Fine-tune clustering parameters in respective modules

## Success Metrics

A successful pipeline execution should achieve:
- ✅ Dataset loaded and validated
- ✅ Classification accuracy > 70%
- ✅ Clustering silhouette score > 0.25
- ✅ All expected output files generated
- ✅ Comprehensive evaluation report created

## Next Steps

After successful pipeline execution:
1. Review the evaluation report in `results/evaluation_report.txt`
2. Examine visualizations in `results/figures/`
3. Use trained models for prediction on new data
4. Consider hyperparameter tuning based on results