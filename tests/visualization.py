import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_topic_sentiments(topic_sentiment):
    """
    Create visualizations for topic-level sentiment analysis
    
    Args:
        topic_sentiment: DataFrame with topic-level sentiment aggregation
    """
    # Filter out outlier topic (-1) for cleaner visualization
    filtered_df = topic_sentiment[topic_sentiment['topic'] != -1].copy()
    
    if len(filtered_df) == 0:
        print("No valid topics to visualize")
        return
        
    # Sort by sentiment score for better visualization
    filtered_df = filtered_df.sort_values('sentiment_score')
    
    # Topic sentiment distribution
    plt.figure(figsize=(12, 8))
    topics = filtered_df['keywords']
    x = np.arange(len(topics))
    width = 0.25
    
    plt.bar(x - width, filtered_df['pct_negative'], width, label='Negative', color='crimson')
    plt.bar(x, filtered_df['pct_neutral'], width, label='Neutral', color='darkgray')
    plt.bar(x + width, filtered_df['pct_positive'], width, label='Positive', color='forestgreen')
    
    plt.xlabel('Topics')
    plt.ylabel('Percentage')
    plt.title('Sentiment Distribution Across Topics')
    plt.xticks(x, topics, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('topic_sentiment_distribution.png')
    
    # Heatmap of topic sentiment scores
    plt.figure(figsize=(10, 6))
    sentiment_pivot = filtered_df[['keywords', 'sentiment_score']].copy()
    sentiment_pivot = sentiment_pivot.set_index('keywords')
    
    sns.heatmap(sentiment_pivot.T, cmap='RdBu_r', center=0, annot=True, 
                cbar_kws={'label': 'Sentiment Score (-1 to 1)'})
    plt.title('Topic Sentiment Heatmap')
    plt.tight_layout()
    plt.savefig('topic_sentiment_heatmap.png')
    
    # Topic size vs sentiment scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(filtered_df['text'], filtered_df['sentiment_score'], 
               s=100, c=filtered_df['sentiment_score'], cmap='RdBu_r')
    
    for i, txt in enumerate(filtered_df['keywords']):
        plt.annotate(txt, (filtered_df['text'].iloc[i], filtered_df['sentiment_score'].iloc[i]),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.colorbar(label='Sentiment Score')
    plt.xlabel('Number of Posts')
    plt.ylabel('Sentiment Score')
    plt.title('Topic Size vs. Sentiment Score')
    plt.tight_layout()
    plt.savefig('topic_size_vs_sentiment.png')
    
    print("Visualizations saved as PNG files")