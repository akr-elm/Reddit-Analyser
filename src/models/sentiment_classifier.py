from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import pandas as pd

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

def perform_sentiment_analysis(texts):
    """
    Perform sentiment analysis on a list of texts using VADER
    
    Args:
        texts: List of text documents
        
    Returns:
        Dictionary with sentiment analysis results
    """
    print(f"Performing sentiment analysis on {len(texts)} documents...")
    
    # Initialize sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    
    # Function to get sentiment scores
    def get_sentiment(text):
        if not isinstance(text, str) or text.strip() == "":
            return {'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0}
        return sia.polarity_scores(text)
    
    # Apply sentiment analysis
    sentiments = [get_sentiment(text) for text in texts]
    
    # Extract sentiment components
    raw_sentiment = [s['compound'] for s in sentiments]
    
    # Categorize sentiment
    def categorize_sentiment(score):
        if score <= -0.05:
            return 0  # Negative
        elif score >= 0.05:
            return 2  # Positive
        else:
            return 1  # Neutral
    
    sentiment_labels = [categorize_sentiment(score) for score in raw_sentiment]
    
    return {
        'raw_sentiment': raw_sentiment,
        'sentiment_score': raw_sentiment,  # Keep compound as the score
        'sentiment_label': sentiment_labels
    }