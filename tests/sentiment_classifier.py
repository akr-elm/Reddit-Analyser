from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
import numpy as np
import os

# Apply the patch for older PyTorch versions
if not hasattr(torch, 'get_default_device'):
    # Monkey patch torch to add the missing function
    def get_default_device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Add the function to the torch module
    torch.get_default_device = get_default_device
    
    # Set environment variable to use our device
    os.environ["PYTORCH_DEVICE"] = str(get_default_device())

def perform_sentiment_analysis(texts):
    """
    Perform sentiment analysis on texts using BERT
    
    Args:
        texts: List of text documents
        
    Returns:
        DataFrame with sentiment labels (0: negative, 1: neutral, 2: positive)
        and confidence scores
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set transformers-specific options to avoid using get_default_device()
    os.environ["TRANSFORMERS_OFFLINE"] = "1"  # This can help avoid some online checks
    
    # Load pre-trained model and tokenizer with device specified directly
    tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    model = BertForSequenceClassification.from_pretrained(
        'nlptown/bert-base-multilingual-uncased-sentiment',
        device_map=None  # Explicitly avoid using device_map feature
    )
    model.to(device)
    model.eval()
    
    results = []
    
    with torch.no_grad():
        for text in texts:
            # Truncate text if needed (BERT has a limit of 512 tokens)
            encoded_input = tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors='pt')
            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
            
            # Get prediction
            output = model(**encoded_input)
            scores = torch.nn.functional.softmax(output.logits, dim=1)
            scores = scores.cpu().numpy()[0]
            
            # Map to sentiment categories (5-class to 3-class)
            # 1,2 -> negative (0), 3 -> neutral (1), 4,5 -> positive (2)
            sentiment_mapping = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2}
            raw_sentiment = np.argmax(scores)
            mapped_sentiment = sentiment_mapping[raw_sentiment]
            
            results.append({
                'sentiment_label': mapped_sentiment,  # 0: negative, 1: neutral, 2: positive
                'sentiment_score': scores[raw_sentiment],
                'sentiment_scores': scores,
                'raw_sentiment': raw_sentiment + 1  # Original 1-5 score
            })
    
    return pd.DataFrame(results)