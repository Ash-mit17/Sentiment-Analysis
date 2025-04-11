import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import MultimodalSentimentModel
import os
import logging
from typing import Dict, List, Tuple, Optional
import json

class TextSentimentAnalyzer:
    def __init__(self):
        """Initialize the text sentiment analyzer."""
        self.model = MultimodalSentimentModel()
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create results directory if it doesn't exist
        if not os.path.exists('analysis_results'):
            os.makedirs('analysis_results')
    
    def analyze_batch(self, data: List[Dict]) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Analyze a batch of text tweets."""
        results = []
        sentiment_counts = {'happy': 0, 'neutral': 0, 'sad': 0}
        
        for tweet in data:
            try:
                tweet_id = tweet.get('id', 'unknown')
                text = tweet.get('text', '')
                
                if not text:
                    self.logger.warning(f"Skipping tweet {tweet_id}: No text content")
                    continue
                
                # Get sentiment scores directly from the model
                pos_score, neg_score, neu_score = self.model.get_text_sentiment_score(text)
                
                # Determine sentiment based on scores
                if pos_score > neg_score and pos_score > neu_score:
                    sentiment = 'happy'
                elif neg_score > pos_score and neg_score > neu_score:
                    sentiment = 'sad'
                else:
                    sentiment = 'neutral'
                
                # Update results
                results.append({
                    'tweet_id': tweet_id,
                    'text': text,
                    'sentiment': sentiment,
                    'positive_score': pos_score,
                    'negative_score': neg_score,
                    'neutral_score': neu_score
                })
                
                # Update sentiment counts
                sentiment_counts[sentiment] += 1
                
            except Exception as e:
                self.logger.error(f"Error processing tweet {tweet.get('id', 'unknown')}: {str(e)}")
                continue
        
        return pd.DataFrame(results), sentiment_counts
    
    def plot_sentiment_distribution(self, sentiment_counts: Dict[str, int], output_path: str):
        """Plot sentiment distribution."""
        sentiments = list(sentiment_counts.keys())
        counts = list(sentiment_counts.values())
        
        plt.figure(figsize=(10, 6))
        plt.bar(sentiments, counts, color=['red', 'gray', 'green'])
        plt.title('Text Sentiment Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.savefig(output_path)
        plt.close()
    
    def save_results(self, results_df: pd.DataFrame, sentiment_counts: Dict[str, int], 
                    prefix: str = 'text_analysis'):
        """Save analysis results."""
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        
        # Save DataFrame to CSV
        csv_path = f'analysis_results/{prefix}_{timestamp}.csv'
        results_df.to_csv(csv_path, index=False)
        
        # Save sentiment distribution plot
        plot_path = f'analysis_results/{prefix}_distribution_{timestamp}.png'
        self.plot_sentiment_distribution(sentiment_counts, plot_path)
        
        # Save summary statistics
        summary = {
            'total_tweets': len(results_df),
            'sentiment_distribution': sentiment_counts
        }
        
        with open(f'analysis_results/{prefix}_summary_{timestamp}.json', 'w') as f:
            json.dump(summary, f, indent=4)
        
        return csv_path, plot_path
    
    def print_summary(self, results_df: pd.DataFrame, sentiment_counts: Dict[str, int]):
        """Print analysis summary."""
        print("\nText Analysis Summary:")
        print(f"Total tweets processed: {len(results_df)}")
        print("\nSentiment Distribution:")
        for sentiment, count in sentiment_counts.items():
            print(f"{sentiment.capitalize()}: {count}")
        
        # Print detailed scores for each tweet
        print("\nDetailed Tweet Analysis:")
        for _, row in results_df.iterrows():
            print(f"\nTweet ID: {row['tweet_id']}")
            print(f"Text: {row['text']}")
            print(f"Sentiment: {row['sentiment']}")
            print(f"Positive Score: {row['positive_score']:.2f}")
            print(f"Negative Score: {row['negative_score']:.2f}")
            print(f"Neutral Score: {row['neutral_score']:.2f}")

def main(input_file: str):
    """Main function to process input file and analyze sentiment."""
    # Read input file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not data:
        print("Error: No valid tweets found in input file.")
        return
    
    # Analyze sentiment
    analyzer = TextSentimentAnalyzer()
    results, sentiment_counts = analyzer.analyze_batch(data)
    
    # Save and display results
    csv_path, plot_path = analyzer.save_results(results, sentiment_counts)
    analyzer.print_summary(results, sentiment_counts)

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python batch_analyzer.py <input_file.json>")
        sys.exit(1)
    
    main(sys.argv[1]) 