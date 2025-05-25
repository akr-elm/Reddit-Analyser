from faker import Faker
import pandas as pd
import random
from datetime import datetime, timedelta
import numpy as np

def generate_synthetic_reddit_data(num_posts=100, political_bias=None):
    """
    Generate synthetic Reddit data for testing the sentiment-topic analysis pipeline
    
    Args:
        num_posts: Number of synthetic posts to generate
        political_bias: Optional bias in the data ('liberal', 'conservative', or None for balanced)
        
    Returns:
        DataFrame with synthetic Reddit data matching the gold table schema
    """
    fake = Faker()
    Faker.seed(42)  # For reproducibility
    
    # Define political subreddits
    liberal_subreddits = ['politics', 'democrats', 'liberal', 'progressive', 'BlueMidterm2022']
    conservative_subreddits = ['Conservative', 'Republican', 'conservatives', 'TheDonald', 'libertarian']
    
    # Common political topics and associated keywords
    political_topics = {
        'healthcare': [
            'Single-payer healthcare is essential for equitable access to medical services.',
            'Government-controlled healthcare leads to inefficiency and poor service.',
            'Medicare for All would reduce costs while improving outcomes for everyone.',
            'Free market solutions for healthcare ensure better quality and innovation.',
            'The current healthcare system needs reform to address rising costs.'
        ],
        'immigration': [
            'We need stronger border security and stricter immigration policies.',
            'Immigration reform should create pathways to citizenship for undocumented residents.',
            'Open borders would harm American workers and strain social services.',
            'Immigrants contribute significantly to our economy and cultural diversity.',
            'DACA recipients deserve legal protection and a path to citizenship.'
        ],
        'economy': [
            'Tax cuts for the wealthy stimulate economic growth and job creation.',
            'Economic inequality is growing and requires government intervention.',
            'Free market capitalism is the most effective economic system.',
            'Corporate tax loopholes should be closed to ensure fair taxation.',
            'Higher minimum wage helps workers while stimulating economic growth.'
        ],
        'climate_change': [
            'Climate change requires immediate action and government regulation.',
            'Environmental regulations hurt businesses and kill jobs unnecessarily.',
            'Renewable energy investment is crucial for our future sustainability.',
            'Market-based solutions are better than government mandates for addressing environmental issues.',
            'The Green New Deal would transform our economy and protect our planet.'
        ],
        'gun_control': [
            'The Second Amendment guarantees the right to bear arms without restriction.',
            'Common-sense gun legislation can prevent mass shootings while respecting rights.',
            'Gun ownership is essential for personal protection and liberty.',
            'Assault weapons and high-capacity magazines should be banned.',
            'Background checks should be required for all gun purchases.'
        ]
    }
    
    # Generate posts
    posts = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    for _ in range(num_posts):
        # Select topic and content based on optional political bias
        topic = random.choice(list(political_topics.keys()))
        
        if political_bias == 'liberal':
            subreddit = random.choice(liberal_subreddits)
            sentiment_bias = random.choice([0, 1, 2, 2, 2])  # Bias toward positive (2)
            text_idx = random.choice([1, 3, 4])  # Liberal-leaning statements
        elif political_bias == 'conservative':
            subreddit = random.choice(conservative_subreddits)
            sentiment_bias = random.choice([0, 1, 2, 2, 2])  # Bias toward positive (2)
            text_idx = random.choice([0, 2, 4])  # Conservative-leaning statements
        else:
            # Balanced approach
            subreddit = random.choice(liberal_subreddits + conservative_subreddits)
            sentiment_bias = random.choice([0, 1, 2])  
            text_idx = random.randint(0, 4)
        
        # Get base content from our predefined topics
        body = political_topics[topic][text_idx]
        
        # Modify content slightly for variety
        words = body.split()
        if len(words) > 10:
            insert_idx = random.randint(5, len(words) - 5)
            words.insert(insert_idx, fake.word())
            body = ' '.join(words)
        
        # Add some random text for uniqueness
        body += ' ' + fake.paragraph(nb_sentences=2)
        
        # Generate title related to the content
        title_words = [w for w in body.split()[:10] if len(w) > 4]
        if title_words:
            title_base = random.choice(title_words).capitalize()
            title = f"{title_base}: {fake.sentence()[:40]}"
        else:
            title = fake.sentence()[:50]
        
        # Generate random date within the past year
        posting_date = fake.date_time_between(start_date=start_date, end_date=end_date).strftime('%Y-%m-%d %H:%M:%S')
        
        # Generate 0-5 comments
        num_comments = random.randint(0, 5)
        comments = [fake.paragraph(nb_sentences=1) for _ in range(num_comments)]
        
        # Determine category based on subreddit
        if subreddit in liberal_subreddits:
            category = 'liberal'
        else:
            category = 'conservative'
        
        posts.append({
            'post_title': title,
            'body': body,
            'posting_date': posting_date,
            'subreddit_name': subreddit,
            'category': category,
            'comments': comments,
            '_sentiment_bias': sentiment_bias,  # Hidden field for validation
            '_topic': topic  # Hidden field for validation
        })
    
    return pd.DataFrame(posts)

def main():
    """
    Generate test data and save it to CSV files
    """
    # Generate balanced dataset
    balanced_data = generate_synthetic_reddit_data(num_posts=200)
    balanced_data.to_csv('synthetic_reddit_balanced.csv', index=False)
    
    # Generate politically biased datasets for testing
    liberal_data = generate_synthetic_reddit_data(num_posts=100, political_bias='liberal')
    liberal_data.to_csv('synthetic_reddit_liberal.csv', index=False)
    
    conservative_data = generate_synthetic_reddit_data(num_posts=100, political_bias='conservative')
    conservative_data.to_csv('synthetic_reddit_conservative.csv', index=False)
    
    print(f"Generated {len(balanced_data)} balanced posts")
    print(f"Generated {len(liberal_data)} liberal-biased posts")
    print(f"Generated {len(conservative_data)} conservative-biased posts")
    
    return balanced_data, liberal_data, conservative_data

if __name__ == "__main__":
    main()