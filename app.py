import streamlit as st
import pandas as pd
import plotly.express as px
import os
import json
import sys

# Set page config - MUST be the first Streamlit command
st.set_page_config(
    page_title="Political Views Analysis",
    page_icon="📊",
    layout="wide"
)

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.pipeline.sentiment_topic_analysis import run_analysis
from src.data.data_loader import load_reddit_data_from_json, format_reddit_data, get_analysis_texts
from src.data.get_reddit_data import fetch_and_save_reddit_data

def load_data(json_path=None, max_records=None):
    """
    Load data from the specified JSON file
    """
    if json_path is None:
        # Use default path
        json_path = os.path.join('data', 'processed', 'corrected_output.json')
    
    # Check if file exists
    if not os.path.exists(json_path):
        st.error(f"JSON file not found: {json_path}")
        return None, None
    
    # Load data
    try:
        st.info(f"Loading data from {json_path}...")
        reddit_data = load_reddit_data_from_json(json_path)
        
        # Apply record limit if specified
        if max_records and len(reddit_data) > max_records:
            reddit_data = reddit_data.iloc[:max_records]
        
        processed_data = format_reddit_data(reddit_data)
        texts = get_analysis_texts(processed_data)
        
        st.success(f"Loaded {len(texts)} posts from {os.path.basename(json_path)}")
        return texts, processed_data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

def plot_topic_sentiment_distribution(topic_analysis):
    """
    Create a visualization of topic sentiment distribution
    """
    # Create a color scale for sentiment scores
    def get_sentiment_color(score):
        if score < -0.25:
            return "crimson"
        elif score > 0.25:
            return "forestgreen"
        else:
            return "darkgray"
    
    # Add sentiment color column
    topic_analysis['sentiment_color'] = topic_analysis['sentiment_score'].apply(get_sentiment_color)
    
    # Sort by sentiment score for better visualization
    sorted_topics = topic_analysis.sort_values('sentiment_score')
    
    # Create bar chart
    fig = px.bar(
        sorted_topics,
        x='keywords', 
        y='sentiment_score',
        color='sentiment_color',
        color_discrete_map="identity",
        labels={'keywords': 'Topics', 'sentiment_score': 'Sentiment Score (-1 to 1)'},
        title='Topic Sentiment Analysis'
    )
    
    # Add horizontal line at y=0
    fig.add_shape(
        type='line',
        x0=0,
        y0=0,
        x1=1,
        y1=0,
        line=dict(color='black', dash='dash'),
        xref='paper',
        yref='y'
    )
    
    return fig

def plot_topic_size_distribution(topic_analysis):
    """
    Create a pie chart showing the distribution of topics by size
    """
    fig = px.pie(
        topic_analysis,
        values='text',
        names='keywords',
        title='Topic Size Distribution'
    )
    
    return fig

def display_topic_details(topic_model, topic_analysis):
    """
    Display information about each topic
    """
    # Sort topics by size
    sorted_topics = topic_analysis.sort_values('text', ascending=False)
    
    for i, (_, topic) in enumerate(sorted_topics.iterrows()):
        topic_id = topic['topic']  # Use the actual topic ID from the data
        st.subheader(f"Topic {topic_id}: {topic['keywords']}")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Topic metadata
            st.markdown(f"**Sentiment Score:** {topic['sentiment_score']:.3f}")
            st.markdown(f"**Posts:** {int(topic['text'])}")
            st.markdown(f"**Sentiment Distribution:**")
            st.markdown(f"- Positive: {topic['pct_positive']:.1f}%")
            st.markdown(f"- Neutral: {topic['pct_neutral']:.1f}%")
            st.markdown(f"- Negative: {topic['pct_negative']:.1f}%")
            
            # Top words - with error handling for LDA model
            try:
                # Get top words using our custom get_topic method
                top_words = topic_model.get_topic(topic_id)
                
                if top_words:
                    st.markdown("**Top Words:**")
                    # Convert to DataFrame for display
                    words_df = pd.DataFrame(top_words[:10], columns=['Word', 'Weight'])
                    st.dataframe(words_df, key=f"words_df_{i}")
                else:
                    st.markdown("**Top Words:** Not available for this topic")
            except Exception as e:
                st.warning(f"Could not display top words: {str(e)}")
        
        with col2:
            # Sentiment distribution pie chart
            topic_sentiment = pd.DataFrame({
                'Sentiment': ['Positive', 'Neutral', 'Negative'],
                'Percentage': [topic['pct_positive'], topic['pct_neutral'], topic['pct_negative']]
            })
            
            fig = px.pie(
                topic_sentiment,
                values='Percentage',
                names='Sentiment',
                color='Sentiment',
                color_discrete_map={
                    'Positive': 'forestgreen',
                    'Neutral': 'darkgray',
                    'Negative': 'crimson'
                }
            )
            # Add a unique key for each topic's pie chart
            st.plotly_chart(fig, use_container_width=True, key=f"sentiment_pie_{i}")
        
        st.markdown("---")

def main():
    # Header
    st.title("Political Views Text Mining")
    st.markdown("Analyze sentiment and topics from political Reddit content")
    
    # Sidebar controls
    st.sidebar.header("Analysis Settings")

    # Data collection section
    with st.expander("Collect Fresh Reddit Data"):
        st.write("Configure Reddit data collection parameters:")
        
        # User inputs
        subreddits = st.text_input("Subreddits (comma-separated)", 
                                "politics,Conservative,liberal,PoliticalHumor")
        limit = st.slider("Posts per subreddit", 5, 100, 20)
        sort_method = st.selectbox("Sort method", 
                                ["hot", "new", "top", "rising", "controversial"])
    
        if st.button("Collect New Data"):
            with st.spinner("Fetching data from Reddit..."):
                # Convert input to list
                subreddit_list = [s.strip() for s in subreddits.split(",")]
                
                # Call our function
                json_path = fetch_and_save_reddit_data(
                    subreddits=subreddit_list,
                    limit=limit,
                    sort_method=sort_method
                )
                
                if json_path:
                    st.success(f"Data collected and saved to {json_path}")
                    st.session_state['data_path'] = json_path
                else:
                    st.error("Failed to collect data")
    
    # File path input
    default_path = os.path.join('data', 'processed', 'reddit_data.json')
    json_path = st.sidebar.text_input("JSON File Path", value=default_path)
    
    # Data limit
    max_records = st.sidebar.slider("Max records to analyze", 100, 5000, 1000)
    
    # Number of topics
    n_topics = st.sidebar.slider("Number of Topics", 3, 15, 8)
    
    # Run analysis button
    run_button = st.sidebar.button("Run Analysis")
    
    if run_button:
        # Load data
        texts, source_data = load_data(json_path, max_records)
        
        if texts:
            # Run analysis
            with st.spinner('Running sentiment and topic analysis with LDA...'):
                post_analysis, topic_analysis, topic_model = run_analysis(
                    texts, 
                    n_topics=n_topics,
                    visualize=False
                )
            
            st.success(f"Analysis complete! Found {len(topic_analysis)} topics from {len(texts)} posts.")
            
            # Display overview metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Posts", len(post_analysis))
            
            with col2:
                avg_sentiment = post_analysis['sentiment_score'].mean()
                st.metric("Average Sentiment", f"{avg_sentiment:.3f}")
            
            with col3:
                num_topics = len(topic_analysis)
                st.metric("Topics Found", num_topics)
            
            # Overview visualizations
            st.header("Topic Overview")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Topic sentiment - add unique key
                fig = plot_topic_sentiment_distribution(topic_analysis)
                st.plotly_chart(fig, use_container_width=True, key="topic_sentiment_overview")
            
            with col2:
                # Topic size - add unique key
                fig = plot_topic_size_distribution(topic_analysis)
                st.plotly_chart(fig, use_container_width=True, key="topic_size_overview")
            
            # Topic details
            st.header("Topic Details")
            display_topic_details(topic_model, topic_analysis)
            
            # Show data samples
            with st.expander("View Data Samples"):
                st.subheader("Topic Analysis Data")
                st.dataframe(topic_analysis, key="topic_analysis_df")
                
                st.subheader("Post Sample (10 records)")
                st.dataframe(post_analysis.head(10), key="post_analysis_df")

if __name__ == "__main__":
    main()