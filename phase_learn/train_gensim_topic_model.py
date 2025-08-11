import os
import json
import pickle
import logging
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import gensim
from gensim import corpora, models
from gensim.models import LdaModel, CoherenceModel
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from config import *
from utils import get_logger, load_ml_dataset, save_model

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

logger = get_logger(__name__)


class GensimTopicModeler:
    def __init__(self, 
                 num_topics: int = 10,
                 alpha: str = 'auto',
                 beta: str = 'auto',
                 passes: int = 10,
                 iterations: int = 100,
                 random_state: int = 42):
        self.num_topics = num_topics
        self.alpha = alpha
        self.beta = beta
        self.passes = passes
        self.iterations = iterations
        self.random_state = random_state
        
        self.dictionary = None
        self.corpus = None
        self.lda_model = None
        self.lemmatizer = WordNetLemmatizer()
        
        # Combine NLTK and Gensim stopwords
        nltk_stops = set(stopwords.words('english'))
        self.stop_words = STOPWORDS.union(nltk_stops)
        
        # Add domain-specific stopwords
        domain_stops = {
            'paper', 'study', 'research', 'method', 'approach', 'result',
            'conclusion', 'introduction', 'abstract', 'figure', 'table',
            'section', 'chapter', 'authors', 'references', 'citation',
            'arxiv', 'preprint', 'published', 'journal', 'conference',
            'algorithm', 'performance', 'evaluation', 'experiment',
            'dataset', 'data', 'model', 'models', 'modeling', 'framework'
        }
        self.stop_words = self.stop_words.union(domain_stops)
    
    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for Gensim topic modeling."""
        # Tokenize and clean
        tokens = simple_preprocess(text, deacc=True, min_len=3, max_len=15)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) 
                 for token in tokens 
                 if token not in self.stop_words and token.isalpha()]
        
        return tokens
    
    def prepare_corpus(self, texts: List[str], min_df: int = 5, max_df: float = 0.8):
        """Prepare corpus and dictionary for Gensim LDA."""
        logger.info(f"Preprocessing {len(texts)} documents...")
        
        # Preprocess all texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Create dictionary
        self.dictionary = corpora.Dictionary(processed_texts)
        
        # Filter extremes
        self.dictionary.filter_extremes(
            no_below=min_df,
            no_above=max_df,
            keep_n=10000
        )
        
        # Create corpus
        self.corpus = [self.dictionary.doc2bow(text) for text in processed_texts]
        
        logger.info(f"Dictionary size: {len(self.dictionary)}")
        logger.info(f"Corpus size: {len(self.corpus)}")
        
        return processed_texts
    
    def train_lda_model(self, texts: List[str]) -> Dict[str, Any]:
        """Train Gensim LDA model."""
        logger.info("Starting Gensim LDA training...")
        
        # Prepare corpus
        processed_texts = self.prepare_corpus(texts)
        
        # Train LDA model
        self.lda_model = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            alpha=self.alpha,
            eta=self.beta,
            passes=self.passes,
            iterations=self.iterations,
            random_state=self.random_state,
            per_word_topics=True,
            eval_every=None
        )
        
        logger.info("LDA training completed")
        
        # Calculate coherence scores
        coherence_scores = self.calculate_coherence_scores(processed_texts)
        
        # Get topic information
        topic_info = self.get_topic_information()
        
        results = {
            'model': self.lda_model,
            'dictionary': self.dictionary,
            'corpus': self.corpus,
            'num_topics': self.num_topics,
            'coherence_scores': coherence_scores,
            'topic_info': topic_info,
            'perplexity': self.lda_model.log_perplexity(self.corpus)
        }
        
        return results
    
    def calculate_coherence_scores(self, processed_texts: List[List[str]]) -> Dict[str, float]:
        """Calculate various coherence scores."""
        logger.info("Calculating coherence scores...")
        
        coherence_scores = {}
        
        # C_v coherence (most commonly used)
        coherence_model_cv = CoherenceModel(
            model=self.lda_model,
            texts=processed_texts,
            dictionary=self.dictionary,
            coherence='c_v'
        )
        coherence_scores['c_v'] = coherence_model_cv.get_coherence()
        
        # U_mass coherence
        coherence_model_umass = CoherenceModel(
            model=self.lda_model,
            corpus=self.corpus,
            dictionary=self.dictionary,
            coherence='u_mass'
        )
        coherence_scores['u_mass'] = coherence_model_umass.get_coherence()
        
        logger.info(f"Coherence C_V: {coherence_scores['c_v']:.4f}")
        logger.info(f"Coherence U_Mass: {coherence_scores['u_mass']:.4f}")
        
        return coherence_scores
    
    def get_topic_information(self) -> List[Dict[str, Any]]:
        """Extract detailed topic information."""
        topics = []
        
        for topic_id in range(self.num_topics):
            # Get top words for topic
            topic_words = self.lda_model.show_topic(topic_id, topn=20)
            
            # Get topic distribution across corpus
            topic_doc_counts = defaultdict(int)
            for doc_topics in self.lda_model.get_document_topics(self.corpus, per_word_topics=False):
                for tid, prob in doc_topics:
                    if tid == topic_id:
                        topic_doc_counts[topic_id] += prob
            
            topic_info = {
                'topic_id': topic_id,
                'words': [(word, float(weight)) for word, weight in topic_words],
                'top_words': [word for word, _ in topic_words[:10]],
                'document_frequency': float(topic_doc_counts[topic_id])
            }
            topics.append(topic_info)
        
        return topics
    
    def predict_document_topics(self, texts: List[str], threshold: float = 0.1) -> List[List[Tuple[int, float]]]:
        """Predict topics for new documents."""
        if not self.lda_model or not self.dictionary:
            raise ValueError("Model must be trained first")
        
        predictions = []
        for text in texts:
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            # Convert to bag of words
            bow = self.dictionary.doc2bow(processed_text)
            
            # Get topic distribution
            doc_topics = self.lda_model.get_document_topics(bow, minimum_probability=threshold)
            
            predictions.append(doc_topics)
        
        return predictions
    
    def save_model(self, model_dir: str):
        """Save the trained model and components."""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save LDA model
        model_path = os.path.join(model_dir, 'gensim_lda_model')
        self.lda_model.save(model_path)
        
        # Save dictionary
        dict_path = os.path.join(model_dir, 'dictionary.dict')
        self.dictionary.save(dict_path)
        
        # Save corpus
        corpus_path = os.path.join(model_dir, 'corpus.mm')
        corpora.MmCorpus.serialize(corpus_path, self.corpus)
        
        # Save model parameters
        params = {
            'num_topics': self.num_topics,
            'alpha': self.alpha,
            'beta': self.beta,
            'passes': self.passes,
            'iterations': self.iterations,
            'random_state': self.random_state
        }
        
        params_path = os.path.join(model_dir, 'model_params.json')
        with open(params_path, 'w') as f:
            json.dump(params, f, indent=2)
        
        logger.info(f"Gensim LDA model saved to {model_dir}")
    
    def load_model(self, model_dir: str):
        """Load a pre-trained model."""
        # Load LDA model
        model_path = os.path.join(model_dir, 'gensim_lda_model')
        self.lda_model = LdaModel.load(model_path)
        
        # Load dictionary
        dict_path = os.path.join(model_dir, 'dictionary.dict')
        self.dictionary = corpora.Dictionary.load(dict_path)
        
        # Load corpus
        corpus_path = os.path.join(model_dir, 'corpus.mm')
        self.corpus = corpora.MmCorpus(corpus_path)
        
        # Load parameters
        params_path = os.path.join(model_dir, 'model_params.json')
        with open(params_path, 'r') as f:
            params = json.load(f)
        
        self.num_topics = params['num_topics']
        self.alpha = params['alpha']
        self.beta = params['beta']
        self.passes = params['passes']
        self.iterations = params['iterations']
        self.random_state = params['random_state']
        
        logger.info(f"Gensim LDA model loaded from {model_dir}")


def train_gensim_topic_model(
    data_path: str = None,
    num_topics: int = None,
    output_dir: str = None
) -> Dict[str, Any]:
    """Train Gensim topic model on paper abstracts."""
    
    # Use config defaults if not provided
    num_topics = num_topics or NUM_TOPICS
    output_dir = output_dir or str(TOPIC_MODEL_DIR)
    data_path = data_path or "../data/metadata/ml_dataset.json"
    
    logger.info(f"Training Gensim LDA with {num_topics} topics")
    
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
    
    logger.info(f"Loaded {len(texts)} text documents")
    
    # Initialize topic modeler
    modeler = GensimTopicModeler(
        num_topics=num_topics,
        alpha='auto',
        beta='auto',
        passes=10,
        iterations=100,
        random_state=RANDOM_STATE
    )
    
    # Train model
    results = modeler.train_lda_model(texts)
    
    # Save model
    gensim_model_dir = os.path.join(output_dir, 'gensim_lda')
    modeler.save_model(gensim_model_dir)
    
    # Save results
    results_path = os.path.join(output_dir, 'gensim_topic_model_results.json')
    results_to_save = {
        'num_topics': results['num_topics'],
        'coherence_scores': results['coherence_scores'],
        'perplexity': float(results['perplexity']),
        'topic_info': results['topic_info']
    }
    
    with open(results_path, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    logger.info(f"Gensim topic modeling results saved to {results_path}")
    
    return results


if __name__ == "__main__":
    results = train_gensim_topic_model()
    
    print(f"\nGensim LDA Training Results:")
    print(f"Number of topics: {results['num_topics']}")
    print(f"Coherence C_V: {results['coherence_scores']['c_v']:.4f}")
    print(f"Coherence U_Mass: {results['coherence_scores']['u_mass']:.4f}")
    print(f"Perplexity: {results['perplexity']:.4f}")
    
    print("\nTop Topics:")
    for topic in results['topic_info'][:5]:
        print(f"Topic {topic['topic_id']}: {', '.join(topic['top_words'][:5])}")