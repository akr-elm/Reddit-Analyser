#!/usr/bin/env python3
import sys
import os
import argparse
import pandas as pd

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.data_loader import load_reddit_data_from_json, format_reddit_data, get_analysis_texts
from src.pipeline.sentiment_topic_analysis import run_analysis
from scripts.generate_test_data import generate_synthetic_reddit_data

def main():
    parser = argparse.ArgumentParser(description="Run sentiment-topic analysis on Reddit data")
    parser.add_argument("--json-file", default=None, help="Path to JSON data file")
    parser.add_argument("--topics", type=int, default=10, help="Number of topics to extract")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    parser.add_argument("--test", action="store_true", help="Use test data instead of JSON file")
    args = parser.parse_args()
    
    if args.test:
        # Use test data
        print("Generating test data...")
        test_data = generate_synthetic_reddit_data(num_posts=200)
        texts = test_data['body'].tolist()
    else:
        # Get data from JSON file
        print("Loading data from JSON file...")
        reddit_data = load_reddit_data_from_json(args.json_file)
        
        if reddit_data.empty:
            print("Error: No data loaded. Exiting.")
            return
        
        # Preprocess the data
        print("Preprocessing data...")
        processed_data = format_reddit_data(reddit_data)
        texts = get_analysis_texts(processed_data)
    
    # Run the analysis
    print(f"Running sentiment-topic analysis with {args.topics} topics on {len(texts)} documents...")
    post_analysis, topic_analysis, topic_model = run_analysis(
        texts, 
        n_topics=args.topics,
        visualize=args.visualize
    )
    
    # Save results
    print("Saving results...")
    os.makedirs('data/processed', exist_ok=True)
    post_analysis.to_csv('data/processed/post_sentiment_results.csv', index=False)
    topic_analysis.to_csv('data/processed/topic_sentiment_results.csv', index=False)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()