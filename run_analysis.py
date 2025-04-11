import json
import pandas as pd
from batch_analyzer import TextSentimentAnalyzer
import logging
import os
from typing import Dict

def analyze_text_file(input_file: str) -> Dict:
    """Analyze text tweets from an input file."""
    try:
        # Read input file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not data:
            return {
                'success': False,
                'input_file': input_file,
                'error': 'Input file is empty'
            }
        
        # Initialize analyzer
        analyzer = TextSentimentAnalyzer()
        
        # Analyze tweets
        results_df, sentiment_counts = analyzer.analyze_batch(data)
        
        if results_df.empty:
            return {
                'success': False,
                'input_file': input_file,
                'error': 'No valid tweets found in the file'
            }
        
        # Save results
        csv_path, plot_path = analyzer.save_results(results_df, sentiment_counts)
        
        # Print summary
        analyzer.print_summary(results_df, sentiment_counts)
        
        return {
            'success': True,
            'input_file': input_file,
            'results_csv': csv_path,
            'distribution_plot': plot_path,
            'total_tweets': len(results_df),
            'sentiment_distribution': sentiment_counts
        }
        
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON file {input_file}: {str(e)}")
        return {
            'success': False,
            'input_file': input_file,
            'error': f'Invalid JSON format: {str(e)}'
        }
    except Exception as e:
        logging.error(f"Error analyzing {input_file}: {str(e)}")
        return {
            'success': False,
            'input_file': input_file,
            'error': str(e)
        }

def main():
    """Main function to run the analysis."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Input file
    text_file = 'text_tweets.json'
    
    # Create results directory if it doesn't exist
    if not os.path.exists('analysis_results'):
        os.makedirs('analysis_results')
    
    # Analyze text tweets
    logger.info("Analyzing text tweets...")
    text_results = analyze_text_file(text_file)
    
    # Print final summary
    print("\nFinal Analysis Summary:")
    print("-" * 30)
    
    if text_results['success']:
        print("\nText Analysis:")
        print(f"Total tweets: {text_results['total_tweets']}")
        print("Sentiment Distribution:")
        for sentiment, count in text_results['sentiment_distribution'].items():
            print(f"  {sentiment.capitalize()}: {count}")
    else:
        print(f"\nText Analysis Error: {text_results['error']}")

if __name__ == '__main__':
    main() 