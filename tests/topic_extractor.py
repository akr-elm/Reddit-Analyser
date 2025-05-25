from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import os

def extract_topics(texts, n_topics=10):
    """
    Extract topics using BERTopic
    
    Args:
        texts: List of text documents
        n_topics: Number of topics to extract
        
    Returns:
        topics: List of topic assignments for each document
        topic_info: DataFrame with topic information
        topic_representations: Dictionary mapping topic IDs to keywords
        topic_model: Trained BERTopic model
    """
    # Make sure NLTK resources are available
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    # Use CountVectorizer for the embedding model to avoid PyTorch compatibility issues
    vectorizer_model = CountVectorizer(stop_words="english")
    
    # Configure BERTopic to use a simpler embedding approach
    topic_model = BERTopic(
        nr_topics=n_topics,
        vectorizer_model=vectorizer_model,
        # Set embedding_model to None to use the default TF-IDF based embeddings
        embedding_model=None,
        verbose=True
    )
    
    print("Extracting topics with BERTopic (TF-IDF mode)...")
    # Fit the model and transform the documents
    topics, probs = topic_model.fit_transform(texts)
    
    # Get topic information
    topic_info = topic_model.get_topic_info()
    
    # Get topic representations (excluding -1 which is the outlier topic)
    topic_ids = [topic for topic in set(topics) if topic != -1]
    topic_representations = {i: topic_model.get_topic(i) for i in topic_ids}
    
    return topics, topic_info, topic_representations, topic_model