import praw
import pandas as pd
from datetime import datetime
import os
import dotenv
import ast
import re
import json

# Load environment variables
dotenv.load_dotenv()
client_id = os.getenv('CLIENT_ID')
client_secret = os.getenv('CLIENT_SECRET')
user_agent = os.getenv('USER_AGENT')    

# === Configure your credentials ===
reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent
)

# === Define subreddits and storage ===
subreddits = ['politics', 'Conservative', 'liberal', 'PoliticalHumor']
limit = 20  # number of posts per subreddit
all_data = []

# === Fetch posts and comments ===
for sub in subreddits:
    subreddit = reddit.subreddit(sub)
    for post in subreddit.hot(limit=limit):
        post.comments.replace_more(limit=0)  # flatten comment tree
        comments = [comment.body for comment in post.comments[:10]]  # top 10 comments
        all_data.append({
            'subreddit': sub,
            'post_id': post.id,
            'title': post.title,
            'selftext': post.selftext,
            'score': post.score,
            'created_utc': datetime.utcfromtimestamp(post.created_utc),
            'num_comments': post.num_comments,
            'comments': comments
        })

# === Convert to DataFrame ===
df = pd.DataFrame(all_data)

def remove_links_from_text(text):
    """
    Remove markdown links from text and keep only the link text
    Example: [mutual trade deal](http://example.com) -> mutual trade deal
    """
    # If text is None or not a string, return empty string
    if not isinstance(text, str):
        return ""
    
    # Pattern to match markdown links: [text](url)
    pattern = r'\[(.*?)\]\(https?://[^\s)]+\)'
    
    # Replace each match with just the link text
    cleaned_text = re.sub(pattern, r'\1', text)
    
    return cleaned_text

def clean_text(text):
    """Enhanced text cleaning to remove noise for better sentiment analysis"""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Remove markdown links but keep the text
    text = remove_links_from_text(text)
    
    # Remove Reddit boilerplate messages (more comprehensive)
    boilerplate_patterns = [
        r"As a reminder.*?civil discussion\.",
        r"In general, be courteous to others.*?permanent ban\.",
        r"If you see comments in violation.*?please report them\.",
        r"For those who have questions.*?outlet criteria\.",
        r"We are actively looking.*?this form\.",
        r"\*\*\*.*?concerns\.\*",
        r"I am a bot.*?automatically\.",
        r"Please \[contact the moderators\].*$",
        r"This thread has been flaired.*$",
        r"This submission was removed.*$"
    ]
    
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, "", text, flags=re.DOTALL|re.IGNORECASE)
    
    # Remove URLs and common internet references
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    text = re.sub(r'\/r\/\w+', '', text)  # Remove subreddit references
    text = re.sub(r'\/u\/\w+', '', text)  # Remove user references
    
    # Remove special characters and formatting
    text = re.sub(r'\[deleted\]|\[removed\]', '', text)
    text = re.sub(r'\*\*|\*|~~|==|__|>', '', text)  # Remove markdown formatting
    text = re.sub(r'&amp;|&lt;|&gt;|&quot;|&nbsp;|&copy;|&reg;', ' ', text)  # HTML entities
    
    # Fix common encoding issues
    text = text.replace('\\n', ' ')
    text = text.replace('\\r', ' ')
    text = text.replace('\\t', ' ')
    text = text.replace('\\"', '"')
    text = text.replace("\\'", "'")
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

# Step 1: Clean title and selftext
df["title"] = df["title"].apply(clean_text)
df["selftext"] = df["selftext"].apply(clean_text)

# Step 2: Parse and clean comments
def parse_and_clean_comments(comment_list):
    """Parse and clean a list of comments"""
    if not isinstance(comment_list, list):
        return []
    
    cleaned_comments = [clean_text(c) for c in comment_list if isinstance(c, str)]
    # Filter out empty comments after cleaning
    cleaned_comments = [c for c in cleaned_comments if c.strip()]
    return cleaned_comments

# Handle the comments properly based on their type
def process_comments(comments_data):
    if isinstance(comments_data, list):
        return parse_and_clean_comments(comments_data)
    elif isinstance(comments_data, str):
        try:
            # If it's stored as a string representation of a list
            comment_list = ast.literal_eval(comments_data)
            return parse_and_clean_comments(comment_list)
        except (ValueError, SyntaxError):
            # If it's just a single comment as string
            return [clean_text(comments_data)] if comments_data.strip() else []
    else:
        return []

df["cleaned_comments"] = df["comments"].apply(process_comments)

# Step 3: Create full text with improved content filtering
def create_full_text(row):
    """Combine title, selftext and top comments into a single text field with noise reduction"""
    title = row["title"].strip()
    selftext = row["selftext"].strip()
    
    # Get up to 3 non-empty comments
    comments = [c for c in row["cleaned_comments"] if len(c.split()) > 3][:3]
    comment_text = " ".join(comments)
    
    # Combine with spaces between components
    components = []
    if title:
        components.append(title)
    if selftext:
        components.append(selftext)
    if comment_text:
        components.append(comment_text)
    
    return " ".join(components)

def fetch_and_save_reddit_data(subreddits=['politics', 'Conservative', 'liberal', 'PoliticalHumor'], 
                               limit=20, 
                               save_path=None,
                               sort_method='hot',
                               comments_limit=10):
    """
    Fetch data from specified subreddits, clean it, and save to JSON
    
    Parameters:
    -----------
    subreddits : list
        List of subreddit names to fetch data from
    limit : int
        Number of posts to fetch from each subreddit
    save_path : str
        Path to save the JSON file (default: "data/processed/reddit_data.json")
    sort_method : str
        Sorting method ('hot', 'new', 'top', 'rising', 'controversial')
    comments_limit : int
        Number of top comments to fetch per post
        
    Returns:
    --------
    str
        Path to the saved JSON file
    """
    import os
    
    # Default save path if not provided
    if save_path is None:
        # Create directories if they don't exist
        os.makedirs("data/processed", exist_ok=True)
        save_path = "data/processed/reddit_data.json"
    
    # Make sure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    all_data = []
    print(f"Fetching data from {len(subreddits)} subreddits: {', '.join(subreddits)}")
    
    # Fetch posts from each subreddit
    for sub in subreddits:
        try:
            print(f"Processing r/{sub}...")
            subreddit = reddit.subreddit(sub)
            
            # Get posts based on the sort method
            if sort_method == 'hot':
                posts = subreddit.hot(limit=limit)
            elif sort_method == 'new':
                posts = subreddit.new(limit=limit)
            elif sort_method == 'top':
                posts = subreddit.top(limit=limit)
            elif sort_method == 'rising':
                posts = subreddit.rising(limit=limit)
            elif sort_method == 'controversial':
                posts = subreddit.controversial(limit=limit)
            else:
                posts = subreddit.hot(limit=limit)
            
            # Process each post
            for post in posts:
                try:
                    post.comments.replace_more(limit=0)  # flatten comment tree
                    comments = [comment.body for comment in post.comments[:comments_limit]]
                    
                    all_data.append({
                        'subreddit': sub,
                        'post_id': post.id,
                        'title': post.title,
                        'selftext': post.selftext,
                        'score': post.score,
                        'created_utc': datetime.utcfromtimestamp(post.created_utc),
                        'num_comments': post.num_comments,
                        'comments': comments
                    })
                except Exception as e:
                    print(f"Error processing post {post.id}: {str(e)}")
                    continue
                
        except Exception as e:
            print(f"Error fetching data from r/{sub}: {str(e)}")
            continue
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    if len(df) == 0:
        print("No data collected. Check your Reddit API credentials and subreddit names.")
        return None
    
    print(f"Collected {len(df)} posts. Cleaning data...")
    
    # Clean data using our existing functions
    df["title"] = df["title"].apply(clean_text)
    df["selftext"] = df["selftext"].apply(clean_text)
    df["cleaned_comments"] = df["comments"].apply(process_comments)
    df["full_text"] = df.apply(create_full_text, axis=1)
    
    # Filter out too short texts
    df["full_text"] = df["full_text"].apply(lambda x: x if len(x.split()) > 5 else "")
    df = df[df["full_text"] != ""]  # Remove empty entries
    
    # Prepare final dataset
    df_cleaned = df[["subreddit", "post_id", "full_text", "score", "created_utc", "num_comments"]]
    
    # Convert datetime to string for JSON serialization
    df_cleaned["created_utc"] = df_cleaned["created_utc"].astype(str)
    
    # Save to JSON
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(df_cleaned.to_dict(orient="records"), f, ensure_ascii=False, indent=2)
    
    print(f"Successfully processed {len(df_cleaned)} posts from {len(subreddits)} subreddits")
    print(f"Data saved to {save_path}")
    
    return save_path
