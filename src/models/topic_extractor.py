from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np

def extract_topics(texts, n_topics=8):
    """
    Extract topics from a collection of texts using LDA
    
    Args:
        texts: List of text documents
        n_topics: Number of topics to extract
        
    Returns:
        Tuple of (topic assignments, topic info, topic representations, topic model)
    """
    print(f"Extracting {n_topics} topics from {len(texts)} documents...")
    
    # Create document-term matrix
    vectorizer = CountVectorizer(
        max_df=0.95,
        min_df=2,
        stop_words='english',
        max_features=1000
    )
    dtm = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    # Fit LDA model
    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=25,
        learning_method='online',
        n_jobs=-1
    )
    lda_model.fit(dtm)
    
    # Assign topics to documents
    doc_topic_dist = lda_model.transform(dtm)
    topics = np.argmax(doc_topic_dist, axis=1)
    topic_probs = np.max(doc_topic_dist, axis=1)
    
    # Extract topic keywords
    def get_topics(model, feature_names, n_top_words=10):
        topics = []
        for topic_idx, topic in enumerate(model.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
            topics.append({
                'Topic': topic_idx,
                'Name': " ".join(top_words[:5]),
                'Keywords': top_words
            })
        return pd.DataFrame(topics)
    
    topic_info = get_topics(lda_model, feature_names)
    
    # Count documents per topic
    topic_counts = pd.Series(topics).value_counts().reset_index()
    topic_counts.columns = ['Topic', 'Count']
    topic_info = topic_info.merge(topic_counts, on='Topic', how='left')
    topic_info['Count'] = topic_info['Count'].fillna(0).astype(int)
    
    # Add get_topic method to match BERTopic API
    def get_topic(topic_id, n_words=10):
        if topic_id < 0 or topic_id >= lda_model.n_components:
            return []
        
        topic = lda_model.components_[topic_id]
        top_word_indices = topic.argsort()[:-n_words-1:-1]
        # Return with weights similar to BERTopic format (word, weight)
        return [(feature_names[i], float(topic[i])) for i in top_word_indices]
    
    # Add the method to the model
    lda_model.get_topic = get_topic
    
    # Create representation similar to BERTopic
    topic_representations = {
        topic_id: get_topic(topic_id, n_words=20)
        for topic_id in range(n_topics)
    }
    
    return topics, topic_info, topic_representations, lda_model