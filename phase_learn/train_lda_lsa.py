import os
import json
import pickle
import logging
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

from utils import get_logger, load_ml_dataset, save_model
from config import *

logger = get_logger(__name__)


class LDATopicModeler:
    """Sklearn-based LDA topic modeling."""
    
    def __init__(self, 
                 num_topics: int = 10,
                 max_features: int = 5000,
                 max_df: float = 0.95,
                 min_df: int = 2,
                 random_state: int = 42):
        self.num_topics = num_topics
        self.max_features = max_features
        self.max_df = max_df
        self.min_df = min_df
        self.random_state = random_state
        
        self.vectorizer = None
        self.lda_model = None
        self.feature_names = None
        
    def fit(self, texts: List[str]):
        """Train LDA model on texts."""
        logger.info(f"Training LDA with {self.num_topics} topics on {len(texts)} documents")
        
        # Adjust parameters for small datasets
        min_df = min(self.min_df, len(texts) // 2) if len(texts) > 1 else 1
        max_df = min(self.max_df, 1.0)
        
        # Vectorize texts
        self.vectorizer = CountVectorizer(
            max_features=self.max_features,
            max_df=max_df,
            min_df=min_df,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        doc_term_matrix = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        # Train LDA
        self.lda_model = LatentDirichletAllocation(
            n_components=self.num_topics,
            random_state=self.random_state,
            max_iter=100,
            learning_method='online',
            batch_size=128,
            evaluate_every=10,
            perp_tol=0.1
        )
        
        self.lda_model.fit(doc_term_matrix)
        
        logger.info("LDA training completed")
        
    def get_topics(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Extract top words for each topic."""
        topics = []
        
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_words_idx = topic.argsort()[-top_n:][::-1]
            top_words = [(self.feature_names[i], float(topic[i])) 
                        for i in top_words_idx]
            
            topics.append({
                'topic_id': topic_idx,
                'words': top_words,
                'top_words': [word for word, _ in top_words]
            })
            
        return topics
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Get topic distributions for texts."""
        doc_term_matrix = self.vectorizer.transform(texts)
        return self.lda_model.transform(doc_term_matrix)
    
    def get_perplexity(self, texts: List[str]) -> float:
        """Calculate perplexity on texts."""
        doc_term_matrix = self.vectorizer.transform(texts)
        return self.lda_model.perplexity(doc_term_matrix)


class LSATopicModeler:
    """LSA (Latent Semantic Analysis) topic modeling using TruncatedSVD."""
    
    def __init__(self, 
                 num_topics: int = 10,
                 max_features: int = 5000,
                 max_df: float = 0.95,
                 min_df: int = 2,
                 random_state: int = 42):
        self.num_topics = num_topics
        self.max_features = max_features
        self.max_df = max_df
        self.min_df = min_df
        self.random_state = random_state
        
        self.pipeline = None
        self.feature_names = None
        self.svd_model = None
        
    def fit(self, texts: List[str]):
        """Train LSA model on texts."""
        logger.info(f"Training LSA with {self.num_topics} topics on {len(texts)} documents")
        
        # Adjust parameters for small datasets
        min_df = min(self.min_df, len(texts) // 2) if len(texts) > 1 else 1
        max_df = min(self.max_df, 1.0)
        
        # Create pipeline with TF-IDF and SVD
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=self.max_features,
                max_df=max_df,
                min_df=min_df,
                stop_words='english',
                ngram_range=(1, 2),
                use_idf=True,
                smooth_idf=True,
                sublinear_tf=True
            )),
            ('svd', TruncatedSVD(
                n_components=self.num_topics,
                random_state=self.random_state,
                algorithm='randomized',
                n_iter=100
            ))
        ])
        
        # Fit pipeline
        self.pipeline.fit(texts)
        
        # Store components
        self.feature_names = self.pipeline['tfidf'].get_feature_names_out()
        self.svd_model = self.pipeline['svd']
        
        logger.info("LSA training completed")
        
    def get_topics(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Extract top words for each topic."""
        topics = []
        
        for topic_idx, topic in enumerate(self.svd_model.components_):
            # Get top positive and negative words
            top_words_idx = np.argsort(np.abs(topic))[-top_n:][::-1]
            top_words = [(self.feature_names[i], float(topic[i])) 
                        for i in top_words_idx]
            
            topics.append({
                'topic_id': topic_idx,
                'words': top_words,
                'top_words': [word for word, _ in top_words]
            })
            
        return topics
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Get topic representations for texts."""
        return self.pipeline.transform(texts)
    
    def get_explained_variance_ratio(self) -> np.ndarray:
        """Get explained variance ratio for each component."""
        return self.svd_model.explained_variance_ratio_


def evaluate_topic_models(lda_model: LDATopicModeler, 
                         lsa_model: LSATopicModeler,
                         texts: List[str]) -> Dict[str, Any]:
    """Evaluate and compare topic models."""
    logger.info("Evaluating topic models...")
    
    evaluation = {}
    
    # LDA evaluation
    lda_perplexity = lda_model.get_perplexity(texts)
    lda_topics = lda_model.get_topics()
    
    evaluation['lda'] = {
        'perplexity': float(lda_perplexity),
        'topics': lda_topics,
        'num_topics': len(lda_topics)
    }
    
    # LSA evaluation
    lsa_explained_variance = lsa_model.get_explained_variance_ratio()
    lsa_topics = lsa_model.get_topics()
    
    evaluation['lsa'] = {
        'explained_variance_ratio': lsa_explained_variance.tolist(),
        'total_explained_variance': float(np.sum(lsa_explained_variance)),
        'topics': lsa_topics,
        'num_topics': len(lsa_topics)
    }
    
    return evaluation


def visualize_topic_comparison(evaluation: Dict[str, Any], output_dir: str):
    """Create visualizations comparing LDA and LSA."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot LSA explained variance
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    explained_var = evaluation['lsa']['explained_variance_ratio']
    plt.plot(range(1, len(explained_var) + 1), explained_var, 'bo-')
    plt.xlabel('Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('LSA Explained Variance by Component')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    cumulative_var = np.cumsum(explained_var)
    plt.plot(range(1, len(cumulative_var) + 1), cumulative_var, 'ro-')
    plt.xlabel('Component')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('LSA Cumulative Explained Variance')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lsa_variance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Topic words comparison
    fig, axes = plt.subplots(2, 1, figsize=(15, 12))
    
    # LDA topics
    lda_topics = evaluation['lda']['topics'][:5]  # Top 5 topics
    for i, topic in enumerate(lda_topics):
        words = [word for word, _ in topic['words'][:8]]
        weights = [weight for _, weight in topic['words'][:8]]
        
        y_pos = np.arange(len(words))
        axes[0].barh(y_pos + i*10, weights, height=0.8, 
                    label=f'Topic {i}', alpha=0.7)
        axes[0].set_yticks(y_pos + i*10)
        axes[0].set_yticklabels(words, fontsize=8)
    
    axes[0].set_xlabel('Weight')
    axes[0].set_title('LDA Topic Words (Top 5 Topics)')
    axes[0].legend()
    
    # LSA topics
    lsa_topics = evaluation['lsa']['topics'][:5]  # Top 5 topics
    for i, topic in enumerate(lsa_topics):
        words = [word for word, _ in topic['words'][:8]]
        weights = [abs(weight) for _, weight in topic['words'][:8]]  # Absolute values for LSA
        
        y_pos = np.arange(len(words))
        axes[1].barh(y_pos + i*10, weights, height=0.8, 
                    label=f'Component {i}', alpha=0.7)
        axes[1].set_yticks(y_pos + i*10)
        axes[1].set_yticklabels(words, fontsize=8)
    
    axes[1].set_xlabel('Absolute Weight')
    axes[1].set_title('LSA Topic Words (Top 5 Components)')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'topic_words_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Visualizations saved to {output_dir}")


def run_lda_lsa_analysis(data_path: str = None, 
                        num_topics: int = None,
                        output_dir: str = None) -> Dict[str, Any]:
    """Run both LDA and LSA topic modeling on existing data."""
    
    # Use config defaults if not provided
    num_topics = num_topics or NUM_TOPICS
    output_dir = output_dir or str(TOPIC_MODEL_DIR)
    data_path = data_path or str(ML_DATASET_FILE)
    
    logger.info("Starting LDA and LSA topic modeling analysis")
    
    # Load data
    data = load_ml_dataset(Path(data_path))
    
    if data is None or 'papers' not in data:
        raise ValueError(f"No paper data found at {data_path}")
    
    # Convert to DataFrame
    papers_df = pd.DataFrame(data['papers'])
    
    if papers_df is None or papers_df.empty:
        raise ValueError(f"No data found at {data_path}")
    
    # Extract text content (use 'text' field instead of 'abstract')
    if 'text' in papers_df.columns:
        texts = papers_df['text'].dropna().tolist()
    elif 'abstract' in papers_df.columns:
        texts = papers_df['abstract'].dropna().tolist()
    else:
        raise ValueError("No 'text' or 'abstract' column found in data")
    
    logger.info(f"Loaded {len(texts)} text documents for analysis")
    
    # Train LDA model
    lda_model = LDATopicModeler(
        num_topics=num_topics,
        random_state=RANDOM_STATE
    )
    lda_model.fit(texts)
    
    # Train LSA model
    lsa_model = LSATopicModeler(
        num_topics=num_topics,
        random_state=RANDOM_STATE
    )
    lsa_model.fit(texts)
    
    # Evaluate models
    evaluation = evaluate_topic_models(lda_model, lsa_model, texts)
    
    # Save models
    models_dir = os.path.join(output_dir, 'lda_lsa_models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Save LDA model
    with open(os.path.join(models_dir, 'lda_model.pkl'), 'wb') as f:
        pickle.dump(lda_model, f)
    
    # Save LSA model
    with open(os.path.join(models_dir, 'lsa_model.pkl'), 'wb') as f:
        pickle.dump(lsa_model, f)
    
    # Save evaluation results
    results_path = os.path.join(output_dir, 'lda_lsa_results.json')
    with open(results_path, 'w') as f:
        json.dump(evaluation, f, indent=2)
    
    # Create visualizations
    viz_dir = os.path.join(output_dir, 'visualizations')
    visualize_topic_comparison(evaluation, viz_dir)
    
    logger.info(f"LDA and LSA analysis completed. Results saved to {output_dir}")
    
    return evaluation


if __name__ == "__main__":
    # Use the correct data path
    data_path = "../data/metadata/ml_dataset.json"
    results = run_lda_lsa_analysis(data_path=data_path)
    
    print("\n" + "="*60)
    print("LDA vs LSA Topic Modeling Results")
    print("="*60)
    
    print(f"\nLDA Results:")
    print(f"Perplexity: {results['lda']['perplexity']:.4f}")
    print(f"Number of topics: {results['lda']['num_topics']}")
    
    print(f"\nLSA Results:")
    print(f"Total explained variance: {results['lsa']['total_explained_variance']:.4f}")
    print(f"Number of components: {results['lsa']['num_topics']}")
    
    print(f"\nTop 3 Topics/Components:")
    print(f"\nLDA Topics:")
    for i, topic in enumerate(results['lda']['topics'][:3]):
        print(f"Topic {i}: {', '.join(topic['top_words'][:5])}")
    
    print(f"\nLSA Components:")
    for i, topic in enumerate(results['lsa']['topics'][:3]):
        print(f"Component {i}: {', '.join(topic['top_words'][:5])}")