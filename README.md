# Multimodal Twitter Sentiment Analysis

This project implements a multimodal sentiment analysis model that combines text and image features to classify tweets into happy, sad, or neutral sentiments.

## Project Structure

- `data_processor.py`: Handles tweet data extraction and preprocessing
- `model.py`: Contains the multimodal sentiment analysis model architecture
- `train.py`: Training script for the model
- `inference.py`: Script for making predictions with the trained model
- `requirements.txt`: Project dependencies

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up Twitter API credentials:
Create a `.env` file with your Twitter API credentials:
```
TWITTER_API_KEY=your_api_key
TWITTER_API_SECRET=your_api_secret
TWITTER_ACCESS_TOKEN=your_access_token
TWITTER_ACCESS_TOKEN_SECRET=your_access_token_secret
```

## Data Collection

Use the `TweetDataProcessor` class in `data_processor.py` to collect and preprocess tweets:

```python
from data_processor import TweetDataProcessor

processor = TweetDataProcessor(api_key, api_secret, access_token, access_token_secret)
tweets = processor.fetch_tweets(query="happy OR sad OR neutral", count=1000)
processor.save_to_csv(tweets, "processed_tweets.csv")
```

## Training

To train the model:

```bash
python train.py
```

The script will:
- Load and preprocess the data
- Train the model
- Save the best model weights
- Generate training curves and confusion matrix plots

## Inference

To analyze sentiment of new tweets:

```python
from inference import SentimentAnalyzer

analyzer = SentimentAnalyzer()
result = analyzer.analyze_sentiment(
    text="Just had the best day ever!",
    image_path="path_to_image.jpg"
)

print(f"Sentiment: {result['sentiment']}")
print(f"Confidence scores: {result['confidence']}")
```

## Model Architecture

The model combines:
- BERT for text feature extraction
- Swin Transformer for image feature extraction
- Fusion layer to combine text and image features
- Classification head for sentiment prediction

## Evaluation Metrics

The model is evaluated using:
- Accuracy
- F1-score
- Confusion matrix

## License

This project is licensed under the MIT License - see the LICENSE file for details. 