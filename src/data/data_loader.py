import pandas as pd
import json
import os
import re

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

def load_reddit_data_from_json(file_path=None):
    """
    Load Reddit data from a JSON file
    
    Parameters:
    file_path (str): Path to the JSON file. If None, uses the default path.
    
    Returns:
    DataFrame: DataFrame containing the Reddit data
    """
    if file_path is None:
        # Use default path relative to project root
        file_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'data', 'raw', 'gold_data.json'
        )
    
    print(f"Loading data from {file_path}...")
    
    try:
        # Load JSON data directly - since we know it's not corrupted
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        print(f"Loaded {len(df)} records from JSON file")
        
        return df
    
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def fix_text_encoding(text):
    """
    Fix common text encoding issues in the dataset
    
    Parameters:
    text (str): The text to fix
    
    Returns:
    str: Fixed text
    """
    if not isinstance(text, str):
        return ""
    
    # First remove markdown links
    text = remove_links_from_text(text)
    
    fixed = text
    
    # Fix spaces that were removed
    fixed = re.sub(r'(\w)([A-Z])', r'\1 \2', fixed)  # Add space between lowercase and uppercase
    fixed = re.sub(r'([.:,;?!])(\w)', r'\1 \2', fixed)  # Add space after punctuation
    
    # Fix common patterns in the text
    fixed = fixed.replace('habeen', 'has been')
    fixed = fixed.replace('iquite', 'is quite')
    fixed = fixed.replace('itrump', 'is trump')
    fixed = fixed.replace('hiattack', 'his attack')
    fixed = fixed.replace('thicould', 'this could')
    fixed = fixed.replace('ithia', 'is this a')
    fixed = fixed.replace('ithi', 'is this')
    fixed = fixed.replace('doeitay', 'does it say')
    fixed = fixed.replace('thiia', 'this is a')
    fixed = fixed.replace('iit', 'is it')
    fixed = fixed.replace('aub', 'sub')
    fixed = fixed.replace('hidaughter', 'his daughter')
    fixed = fixed.replace('treatedo', 'treated so')
    fixed = fixed.replace('tore', 'store')
    fixed = fixed.replace('here\'the', 'here\'s the')
    fixed = fixed.replace('que tion', 'question')
    fixed = fixed.replace('po t', 'post')
    fixed = fixed.replace('tarrif', 'tariff')
    fixed = fixed.replace('thi ', 'this ')
    
    # Fix newlines and unicode escape sequences
    fixed = fixed.replace('\\n', '\n')
    fixed = re.sub(r'\s+', ' ', fixed)  # Normalize whitespace
    
    # Remove any remaining URL fragments
    fixed = re.sub(r'https?://\S+', '', fixed)
    fixed = re.sub(r'www\.\S+', '', fixed)
    
    return fixed.strip()

def format_reddit_data(df):
    """
    Format the Reddit data for analysis
    
    Parameters:
    df (DataFrame): DataFrame containing the Reddit data
    
    Returns:
    DataFrame: Formatted DataFrame ready for sentiment analysis and topic modeling
    """
    if df.empty:
        print("Warning: Empty DataFrame, nothing to format")
        return df
    
    # Create a copy to avoid modifying the original
    formatted_df = df.copy()
    
    # Fix text encoding issues in key text fields
    print("Cleaning and preprocessing text...")
    if 'body' in formatted_df.columns:
        formatted_df['body'] = formatted_df['body'].apply(fix_text_encoding)
    
    if 'post_title' in formatted_df.columns:
        formatted_df['post_title'] = formatted_df['post_title'].apply(fix_text_encoding)
    
    if 'comment' in formatted_df.columns:
        formatted_df['comment'] = formatted_df['comment'].apply(fix_text_encoding)
    
    # Combine title, body, and comment for text analysis
    print("Creating full text field for analysis...")
    if 'post_title' in formatted_df.columns and 'body' in formatted_df.columns:
        formatted_df['full_text'] = formatted_df['post_title'].fillna('') + " " + formatted_df['body'].fillna('')
        
        # Add comments to full text if available
        if 'comment' in formatted_df.columns:
            formatted_df['full_text'] = formatted_df['full_text'] + " " + formatted_df['comment'].fillna('')
    elif 'body' in formatted_df.columns:
        formatted_df['full_text'] = formatted_df['body'].fillna('')
    elif 'post_title' in formatted_df.columns:
        formatted_df['full_text'] = formatted_df['post_title'].fillna('')
    
    # Create a datetime column if posting_date is available
    if 'posting_date' in formatted_df.columns:
        try:
            formatted_df['posting_datetime'] = pd.to_datetime(formatted_df['posting_date'])
        except:
            # Keep as string if conversion fails
            formatted_df['posting_datetime'] = formatted_df['posting_date']
    
    # Select relevant columns
    columns_to_keep = [col for col in 
                      ['id', 'post_title', 'body', 'full_text', 'posting_date', 
                       'subreddit_name', 'category', 'comment'] 
                      if col in formatted_df.columns]
    
    result_df = formatted_df[columns_to_keep].copy()
    
    # Remove NaN values
    result_df = result_df.fillna('')
    
    return result_df

def get_analysis_texts(df):
    """
    Extract the text content for analysis
    
    Parameters:
    df (DataFrame): Formatted DataFrame from format_reddit_data
    
    Returns:
    list: List of text strings for analysis
    """
    if 'full_text' in df.columns:
        return df['full_text'].tolist()
    elif 'body' in df.columns:
        return df['body'].tolist()
    elif 'post_title' in df.columns:
        return df['post_title'].tolist()
    else:
        print("Warning: No suitable text column found for analysis")
        return []