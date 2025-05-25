from src.models.sentiment_classifier import perform_sentiment_analysis
from src.models.topic_extractor import extract_topics
from src.visualization.visualizer import visualize_topic_sentiments
import pandas as pd
from collections import Counter

def hybrid_sentiment_topic_analysis(texts, n_topics=10):
    """
    Perform hybrid sentiment analysis and topic modeling
    
    Args:
        texts: List of text documents
        n_topics: Number of topics to extract
        
    Returns:
        DataFrame with per-post analysis and topic-level aggregation
    """
    # Extract topics
    topics, topic_info, topic_representations, topic_model = extract_topics(texts, n_topics)
    
    # Perform sentiment analysis
    sentiment_results = perform_sentiment_analysis(texts)
    
    # Create a DataFrame with texts, topics, and sentiment
    df = pd.DataFrame({
        'text': texts,
        'topic': topics,
        'sentiment_label': sentiment_results['sentiment_label'],
        'sentiment_score': sentiment_results['sentiment_score'],
        'raw_sentiment': sentiment_results['raw_sentiment']
    })
    
    # Aggregate sentiment by topic
    topic_sentiment = df.groupby('topic').agg({
        'sentiment_label': lambda x: Counter(x),
        'raw_sentiment': 'mean',
        'text': 'count'
    }).reset_index()
    
    # Calculate percentage of positive, neutral, and negative for each topic
    for topic_idx, row in topic_sentiment.iterrows():
        counter = row['sentiment_label']
        total = sum(counter.values())
        if total > 0:
            topic_sentiment.at[topic_idx, 'pct_negative'] = counter.get(0, 0) / total * 100
            topic_sentiment.at[topic_idx, 'pct_neutral'] = counter.get(1, 0) / total * 100
            topic_sentiment.at[topic_idx, 'pct_positive'] = counter.get(2, 0) / total * 100
        else:
            topic_sentiment.at[topic_idx, 'pct_negative'] = 0
            topic_sentiment.at[topic_idx, 'pct_neutral'] = 0
            topic_sentiment.at[topic_idx, 'pct_positive'] = 0
        
        # Add topic keywords
        if row['topic'] != -1:  # Skip outlier topic (-1)
            topic_keywords = [word for word, _ in topic_model.get_topic(row['topic'])]
            topic_sentiment.at[topic_idx, 'keywords'] = ', '.join(topic_keywords[:5])
        else:
            topic_sentiment.at[topic_idx, 'keywords'] = 'Outliers'
    
    # Calculate overall sentiment score per topic (-1 to 1 range)
    topic_sentiment['sentiment_score'] = (topic_sentiment['pct_positive'] - topic_sentiment['pct_negative']) / 100
    
    return df, topic_sentiment, topic_model

def run_analysis(texts, n_topics=10, visualize=True):
    """
    Run the complete hybrid sentiment-topic analysis pipeline
    """
    post_analysis, topic_analysis, topic_model = hybrid_sentiment_topic_analysis(texts, n_topics)
    
    print(f"Analyzed {len(texts)} posts across {len(topic_analysis)} topics")
    print("\nTopic-level Sentiment Summary:")
    print(topic_analysis[['topic', 'keywords', 'text', 'sentiment_score', 
                         'pct_positive', 'pct_neutral', 'pct_negative']].sort_values('sentiment_score', ascending=False))
    
    if visualize:
        visualize_topic_sentiments(topic_analysis)
        
    return post_analysis, topic_analysis, topic_model