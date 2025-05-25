
**RedditAnalyser** is a tool for analyzing political discourse on Reddit using natural language processing techniques. It combines topic modeling with sentiment analysis to uncover trends and patterns in political discussions across different subreddits.

## Features

- **Topic Modeling**: Uses Latent Dirichlet Allocation (LDA) to discover topics within political discussions
- **Sentiment Analysis**: Employs VADER sentiment analysis to gauge the emotional tone of content
- **Interactive Visualization**: Provides interactive visualizations for exploring topics and sentiment patterns
- **Cross-Subreddit Analysis**: Compare discourse patterns across political subreddits
- **Streamlit Web Interface**: User-friendly interface for analyzing data without coding

## Installation

```bash
# Clone the repository
git clone https://github.com/akr-elm/Reddit-Analyser.git
cd Reddit-Analyser

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Streamlit App

```bash
streamlit run app.py
```

This will launch the web interface where you can:
1. Collect fresh data from Reddit
2. Analyze existing Reddit data
3. Explore topics and sentiment patterns
4. View detailed breakdowns of political discourse

## Project Structure

```
politisense/
├── app.py                # Streamlit application
├── data/                 # Data directory
│   ├── processed/        # Processed data
│   └── raw/              # Raw data
├── notebooks/            # Jupyter notebooks for exploration
├── src/                  # Source code
│   ├── data/             # Data loading and processing
│   ├── models/           # Topic and sentiment models
│   └── pipeline/         # Analysis pipeline
└── requirements.txt      # Python dependencies
```

## Technology Stack

- **Python 3.8+**
- **scikit-learn**: For LDA topic modeling
- **NLTK**: For VADER sentiment analysis
- **Pandas**: For data manipulation
- **Plotly & Matplotlib**: For data visualization
- **Streamlit**: For the web interface

## Data Sources

The application analyzes data from political subreddits by default including:
- r/politics
- r/Conservative
- r/liberal
- r/PoliticalHumor
Other subreddits could be specified

## Future Improvements

- Add time-series analysis to track opinion shifts over time
- Implement more advanced NLP models for better topic coherence
- Add cross-platform analysis (Twitter, Facebook, etc.)
- Improve visualization capabilities

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Reddit API for providing access to discussion data
- The VADER sentiment analysis tool
- The scikit-learn library for LDA implementation

---

*Note: This project is for research and educational purposes. Please adhere to Reddit's API terms of service when collecting data.*
