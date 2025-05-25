from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import pandas as pd


def connect_to_cassandra(host='cassandra', port=9042, username=None, password=None):
    """
    Connect to Cassandra database
    """
    if username and password:
        auth_provider = PlainTextAuthProvider(username=username, password=password)
        cluster = Cluster([host], port=port, auth_provider=auth_provider)
    else:
        cluster = Cluster([host], port=port)
    
    session = cluster.connect()
    return session

def get_reddit_data(session, keyspace, table):
    """
    Retrieve data from Cassandra
    """
    session.set_keyspace(keyspace)
    query = f"SELECT * FROM {table}"
    rows = session.execute(query)
    
    # Convert to pandas DataFrame for easier processing
    df = pd.DataFrame(list(rows))
    return df

def format_reddit_data_gold(df):
    """
    Format the Reddit data from the gold table for analysis
    
    Parameters:
    df (DataFrame): DataFrame containing data from the reddit_data.gold table
    
    Returns:
    DataFrame: Formatted DataFrame ready for sentiment analysis and topic modeling
    """
    # Combine title and body for text analysis
    df['full_text'] = df['post_title'] + " " + df['body']
    
    # Convert comments list to a single string if needed
    if 'comments' in df.columns:
        df['comments_text'] = df['comments'].apply(lambda x: " ".join(x) if x else "")
        # Add comments to full text for comprehensive analysis
        df['full_text'] = df['full_text'] + " " + df['comments_text']
    
    # Create a datetime column if posting_date is a string
    if 'posting_date' in df.columns:
        try:
            df['posting_datetime'] = pd.to_datetime(df['posting_date'])
        except:
            # Keep as string if conversion fails
            df['posting_datetime'] = df['posting_date']
    
    # Create a clean DataFrame with all relevant columns
    formatted_df = df[['post_title', 'body', 'full_text', 'posting_date', 
                       'subreddit_name', 'category']].copy()
    
    # Add any additional preprocessing steps
    # Remove NaN values
    formatted_df = formatted_df.fillna('')
    
    return formatted_df