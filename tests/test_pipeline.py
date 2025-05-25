import pandas as pd
from sentiment_topic_analysis import run_analysis
from generate_test_data import generate_synthetic_reddit_data
import matplotlib.pyplot as plt
import seaborn as sns

def test_with_synthetic_data(dataset_type='balanced', num_posts=100, n_topics=5):
    """
    Test the sentiment-topic analysis pipeline with synthetic data
    
    Args:
        dataset_type: Type of dataset to generate ('balanced', 'liberal', 'conservative')
        num_posts: Number of posts to generate
        n_topics: Number of topics to extract
    """
    print(f"Generating {dataset_type} dataset with {num_posts} posts...")
    
    # Generate data
    if dataset_type == 'liberal':
        data = generate_synthetic_reddit_data(num_posts=num_posts, political_bias='liberal')
    elif dataset_type == 'conservative':
        data = generate_synthetic_reddit_data(num_posts=num_posts, political_bias='conservative')
    else:
        data = generate_synthetic_reddit_data(num_posts=num_posts)
    
    # Extract text for analysis
    texts = data['body'].tolist()
    
    print(f"Running sentiment-topic analysis with {n_topics} topics...")
    
    # Run the analysis
    post_analysis, topic_analysis, topic_model = run_analysis(texts, n_topics=n_topics)
    
    # Print some statistics
    print(f"\nSentiment distribution in the dataset:")
    sentiment_counts = post_analysis['sentiment_label'].value_counts()
    print(f"Negative: {sentiment_counts.get(0, 0)} ({sentiment_counts.get(0, 0)/len(post_analysis)*100:.1f}%)")
    print(f"Neutral: {sentiment_counts.get(1, 0)} ({sentiment_counts.get(1, 0)/len(post_analysis)*100:.1f}%)")
    print(f"Positive: {sentiment_counts.get(2, 0)} ({sentiment_counts.get(2, 0)/len(post_analysis)*100:.1f}%)")
    
    # Create validation visualization comparing our hidden sentiment bias with detected sentiment
    if '_sentiment_bias' in data.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Join the hidden bias with detected sentiment
        validation_df = pd.DataFrame({
            'Hidden Bias': data['_sentiment_bias'],
            'Detected Sentiment': post_analysis['sentiment_label']
        })
        
        # Create confusion matrix
        confusion = pd.crosstab(
            validation_df['Hidden Bias'], 
            validation_df['Detected Sentiment'],
            rownames=['Hidden'], 
            colnames=['Detected'],
            normalize='index'
        )
        
        sns.heatmap(confusion, annot=True, fmt='.2f', cmap='Blues', ax=ax)
        plt.title('Hidden vs. Detected Sentiment')
        plt.savefig(f'sentiment_validation_{dataset_type}.png')
        
        print("\nSentiment validation results saved to sentiment_validation.png")
    
    # Return all results for further inspection if needed
    return data, post_analysis, topic_analysis, topic_model

if __name__ == "__main__":
    # Test with different datasets
    print("=== Testing with balanced dataset ===")
    balanced_results = test_with_synthetic_data('balanced', num_posts=200, n_topics=5)
    
    print("\n=== Testing with liberal-biased dataset ===")
    liberal_results = test_with_synthetic_data('liberal', num_posts=100, n_topics=5)
    
    print("\n=== Testing with conservative-biased dataset ===")
    conservative_results = test_with_synthetic_data('conservative', num_posts=100, n_topics=5)