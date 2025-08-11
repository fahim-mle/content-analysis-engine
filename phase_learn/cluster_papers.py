# phase_learn/cluster_papers.py
import json
import gzip
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hardware-aware settings
BATCH_SIZE = 32  # For sentence transformers
MAX_FEATURES = 5000  # TF-IDF features
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 dimension


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


def prepare_texts_for_clustering(papers: List[Dict]) -> Tuple[List[str], List[Dict]]:
    """Prepare texts and metadata for clustering."""
    logger.info("Preparing texts for clustering...")
    
    texts = []
    metadata = []
    
    for paper in papers:
        # Combine title and abstract/first part of text
        title = paper.get('title', '')
        text = paper.get('text', '')
        
        # Use title + first 500 chars for clustering (memory efficient)
        combined_text = f"{title}. {text[:500]}"
        texts.append(combined_text)
        
        metadata.append({
            'arxiv_id': paper.get('arxiv_id', ''),
            'title': title,
            'categories': paper.get('categories', []),
            'quality_score': paper.get('quality_score', 0.0)
        })
    
    logger.info(f"Prepared {len(texts)} texts for clustering")
    return texts, metadata


def create_tfidf_embeddings(texts: List[str]) -> np.ndarray:
    """Create TF-IDF embeddings for clustering."""
    logger.info("Creating TF-IDF embeddings...")
    
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        lowercase=True
    )
    
    embeddings = vectorizer.fit_transform(texts)
    logger.info(f"TF-IDF embeddings shape: {embeddings.shape}")
    
    return embeddings.toarray(), vectorizer


def create_sentence_embeddings(texts: List[str]) -> np.ndarray:
    """Create sentence embeddings using SentenceTransformers (hardware-aware)."""
    logger.info("Creating sentence embeddings with all-MiniLM-L6-v2...")
    
    # Use lightweight model suitable for 4GB GPU
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Process in batches to manage memory
    embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        batch_embeddings = model.encode(batch, batch_size=BATCH_SIZE)
        embeddings.extend(batch_embeddings)
        
        if i % (BATCH_SIZE * 10) == 0:
            logger.info(f"Processed {i + len(batch)}/{len(texts)} texts")
    
    embeddings = np.array(embeddings)
    logger.info(f"Sentence embeddings shape: {embeddings.shape}")
    
    return embeddings


def perform_kmeans_clustering(embeddings: np.ndarray, k_range: range = range(3, 15)) -> Dict:
    """Perform K-Means clustering with different k values."""
    logger.info(f"Performing K-Means clustering for k in {k_range}")
    
    results = {}
    best_k = None
    best_score = -1
    
    for k in k_range:
        if k >= len(embeddings):
            logger.warning(f"Skipping k={k}, not enough samples")
            continue
            
        # Fit K-Means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Calculate metrics
        silhouette = silhouette_score(embeddings, cluster_labels)
        calinski = calinski_harabasz_score(embeddings, cluster_labels)
        
        results[k] = {
            'model': kmeans,
            'labels': cluster_labels,
            'silhouette_score': silhouette,
            'calinski_harabasz_score': calinski,
            'inertia': kmeans.inertia_
        }
        
        logger.info(f"K={k}: Silhouette={silhouette:.3f}, Calinski-Harabasz={calinski:.1f}")
        
        # Track best based on silhouette score
        if silhouette > best_score:
            best_score = silhouette
            best_k = k
    
    logger.info(f"Best K-Means: k={best_k} with silhouette score {best_score:.3f}")
    
    return {
        'results': results,
        'best_k': best_k,
        'best_model': results[best_k]['model'] if best_k else None,
        'best_labels': results[best_k]['labels'] if best_k else None
    }


def perform_dbscan_clustering(embeddings: np.ndarray) -> Dict:
    """Perform DBSCAN clustering with automatic parameter selection."""
    logger.info("Performing DBSCAN clustering...")
    
    # Try different eps values
    eps_values = [0.1, 0.3, 0.5, 0.7, 1.0]
    best_result = None
    best_score = -1
    
    for eps in eps_values:
        dbscan = DBSCAN(eps=eps, min_samples=3)
        cluster_labels = dbscan.fit_predict(embeddings)
        
        # Skip if all points are noise or all in one cluster
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        if n_clusters < 2 or n_clusters == 1:
            continue
            
        # Calculate silhouette score (exclude noise points)
        if n_clusters > 1 and len(set(cluster_labels)) > 1:
            valid_indices = cluster_labels != -1
            if valid_indices.sum() > 1:
                silhouette = silhouette_score(
                    embeddings[valid_indices], 
                    cluster_labels[valid_indices]
                )
            else:
                silhouette = -1
        else:
            silhouette = -1
        
        logger.info(f"DBSCAN eps={eps}: {n_clusters} clusters, {n_noise} noise points, silhouette={silhouette:.3f}")
        
        if silhouette > best_score:
            best_score = silhouette
            best_result = {
                'model': dbscan,
                'labels': cluster_labels,
                'eps': eps,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'silhouette_score': silhouette
            }
    
    return best_result


def analyze_clusters(labels: np.ndarray, metadata: List[Dict]) -> Dict:
    """Analyze cluster composition and characteristics."""
    logger.info("Analyzing cluster composition...")
    
    analysis = {}
    unique_labels = set(labels)
    
    for label in unique_labels:
        if label == -1:  # Noise in DBSCAN
            cluster_name = 'noise'
        else:
            cluster_name = f'cluster_{label}'
        
        # Get papers in this cluster
        cluster_indices = np.where(labels == label)[0]
        cluster_papers = [metadata[i] for i in cluster_indices]
        
        # Analyze categories
        all_categories = []
        for paper in cluster_papers:
            all_categories.extend(paper['categories'])
        
        from collections import Counter
        category_counts = Counter(all_categories)
        
        # Calculate quality score statistics
        quality_scores = [paper['quality_score'] for paper in cluster_papers]
        
        analysis[cluster_name] = {
            'size': len(cluster_papers),
            'sample_titles': [p['title'] for p in cluster_papers[:3]],
            'top_categories': dict(category_counts.most_common(5)),
            'avg_quality_score': np.mean(quality_scores) if quality_scores else 0,
            'paper_ids': [p['arxiv_id'] for p in cluster_papers[:10]]  # First 10 IDs
        }
    
    return analysis


def visualize_clusters(embeddings: np.ndarray, labels: np.ndarray, title: str, output_path: Path):
    """Create 2D visualization of clusters using PCA."""
    logger.info(f"Creating cluster visualization: {title}")
    
    # Reduce to 2D using PCA
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    unique_labels = set(labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        if label == -1:
            # Noise points in black
            color = 'black'
            marker = 'x'
            label_name = 'Noise'
        else:
            marker = 'o'
            label_name = f'Cluster {label}'
        
        mask = labels == label
        plt.scatter(
            embeddings_2d[mask, 0], 
            embeddings_2d[mask, 1],
            c=[color], 
            marker=marker,
            s=50,
            alpha=0.7,
            label=f'{label_name} ({mask.sum()})'
        )
    
    plt.title(title)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Visualization saved to {output_path}")


def save_clustering_results(results: Dict, output_path: Path):
    """Save clustering results to file."""
    logger.info(f"Saving clustering results to {output_path}")
    
    # Prepare serializable results
    serializable_results = {}
    for method, result in results.items():
        if result is None:
            serializable_results[method] = None
            continue
            
        serializable_result = {}
        for key, value in result.items():
            if key == 'model':
                # Save model separately
                continue
            elif isinstance(value, np.ndarray):
                serializable_result[key] = value.tolist()
            elif isinstance(value, dict):
                # Handle nested results
                nested_dict = {}
                for k, v in value.items():
                    if isinstance(v, dict) and 'model' in v:
                        # Skip model objects, keep metrics
                        model_free = {k2: v2 for k2, v2 in v.items() if k2 != 'model'}
                        nested_dict[k] = model_free
                    elif isinstance(v, np.ndarray):
                        nested_dict[k] = v.tolist()
                    else:
                        nested_dict[k] = v
                serializable_result[key] = nested_dict
            else:
                serializable_result[key] = value
        
        serializable_results[method] = serializable_result
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)


def main():
    """Main function to perform paper clustering analysis."""
    logger.info("Starting paper clustering analysis...")
    
    # Paths
    data_path = Path("data/metadata/ml_dataset.json.gz")
    output_dir = Path("data/models/clustering")
    figures_dir = Path("results/figures")
    
    # Load dataset
    papers = load_ml_dataset(data_path)
    if not papers:
        logger.error("No papers loaded. Exiting.")
        return
    
    # Prepare texts
    texts, metadata = prepare_texts_for_clustering(papers)
    if len(texts) < 10:
        logger.error("Insufficient data for clustering. Exiting.")
        return
    
    # Create embeddings (both TF-IDF and sentence embeddings)
    logger.info("Creating embeddings...")
    
    # TF-IDF embeddings
    tfidf_embeddings, tfidf_vectorizer = create_tfidf_embeddings(texts)
    
    # Sentence embeddings (lightweight model)
    try:
        sentence_embeddings = create_sentence_embeddings(texts)
    except Exception as e:
        logger.warning(f"Failed to create sentence embeddings: {e}")
        sentence_embeddings = None
    
    # Clustering results
    results = {}
    
    # K-Means on TF-IDF
    logger.info("Performing K-Means clustering on TF-IDF embeddings...")
    kmeans_tfidf = perform_kmeans_clustering(tfidf_embeddings)
    results['kmeans_tfidf'] = kmeans_tfidf
    
    # K-Means on sentence embeddings (if available)
    if sentence_embeddings is not None:
        logger.info("Performing K-Means clustering on sentence embeddings...")
        kmeans_sentence = perform_kmeans_clustering(sentence_embeddings)
        results['kmeans_sentence'] = kmeans_sentence
    
    # DBSCAN on TF-IDF
    logger.info("Performing DBSCAN clustering on TF-IDF embeddings...")
    dbscan_tfidf = perform_dbscan_clustering(tfidf_embeddings)
    results['dbscan_tfidf'] = dbscan_tfidf
    
    # Analyze best clustering results
    analyses = {}
    
    if kmeans_tfidf['best_labels'] is not None:
        analyses['kmeans_tfidf'] = analyze_clusters(kmeans_tfidf['best_labels'], metadata)
        
        # Visualize K-Means TF-IDF
        visualize_clusters(
            tfidf_embeddings, 
            kmeans_tfidf['best_labels'],
            f"K-Means Clustering (TF-IDF, k={kmeans_tfidf['best_k']})",
            figures_dir / "kmeans_tfidf_clusters.png"
        )
    
    if sentence_embeddings is not None and 'kmeans_sentence' in results:
        if results['kmeans_sentence']['best_labels'] is not None:
            analyses['kmeans_sentence'] = analyze_clusters(results['kmeans_sentence']['best_labels'], metadata)
            
            # Visualize K-Means Sentence
            visualize_clusters(
                sentence_embeddings,
                results['kmeans_sentence']['best_labels'],
                f"K-Means Clustering (Sentence Embeddings, k={results['kmeans_sentence']['best_k']})",
                figures_dir / "kmeans_sentence_clusters.png"
            )
    
    if dbscan_tfidf is not None:
        analyses['dbscan_tfidf'] = analyze_clusters(dbscan_tfidf['labels'], metadata)
        
        # Visualize DBSCAN
        visualize_clusters(
            tfidf_embeddings,
            dbscan_tfidf['labels'],
            f"DBSCAN Clustering (TF-IDF, eps={dbscan_tfidf['eps']})",
            figures_dir / "dbscan_tfidf_clusters.png"
        )
    
    # Save results
    save_clustering_results(results, output_dir / "clustering_results.json")
    
    # Save cluster analyses
    analyses_file = output_dir / "cluster_analyses.json"
    with open(analyses_file, 'w') as f:
        json.dump(analyses, f, indent=2, default=str)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("CLUSTERING ANALYSIS SUMMARY")
    logger.info("="*60)
    
    if kmeans_tfidf['best_k']:
        logger.info(f"Best K-Means (TF-IDF): k={kmeans_tfidf['best_k']}")
        best_result = kmeans_tfidf['results'][kmeans_tfidf['best_k']]
        logger.info(f"  Silhouette Score: {best_result['silhouette_score']:.3f}")
        logger.info(f"  Calinski-Harabasz: {best_result['calinski_harabasz_score']:.1f}")
    
    if sentence_embeddings is not None and 'kmeans_sentence' in results:
        best_k_sent = results['kmeans_sentence']['best_k']
        if best_k_sent:
            logger.info(f"Best K-Means (Sentence): k={best_k_sent}")
            best_result_sent = results['kmeans_sentence']['results'][best_k_sent]
            logger.info(f"  Silhouette Score: {best_result_sent['silhouette_score']:.3f}")
    
    if dbscan_tfidf:
        logger.info(f"DBSCAN (TF-IDF): eps={dbscan_tfidf['eps']}")
        logger.info(f"  Clusters: {dbscan_tfidf['n_clusters']}")
        logger.info(f"  Noise points: {dbscan_tfidf['n_noise']}")
        logger.info(f"  Silhouette Score: {dbscan_tfidf['silhouette_score']:.3f}")
    
    logger.info("\nOutputs saved to:")
    logger.info(f"- Results: {output_dir}/clustering_results.json")
    logger.info(f"- Analyses: {output_dir}/cluster_analyses.json")
    logger.info(f"- Figures: {figures_dir}/")
    
    logger.info("Paper clustering analysis completed.")


if __name__ == "__main__":
    main()