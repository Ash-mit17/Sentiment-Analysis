import os
import re
import tweepy
from PIL import Image
import pandas as pd
from typing import List, Dict, Tuple
import requests
from io import BytesIO
import logging

class TweetDataProcessor:
    def __init__(self, api_key: str, api_secret: str, access_token: str, access_token_secret: str):
        """Initialize the TweetDataProcessor with Twitter API credentials."""
        self.auth = tweepy.OAuthHandler(api_key, api_secret)
        self.auth.set_access_token(access_token, access_token_secret)
        self.api = tweepy.API(self.auth)
        self.media_dir = "media"
        os.makedirs(self.media_dir, exist_ok=True)

    def clean_text(self, text: str) -> str:
        """Clean tweet text by removing mentions, hashtags, URLs, and special characters."""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        # Remove emojis and special characters
        text = re.sub(r'[^\w\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text

    def download_media(self, media_url: str, tweet_id: str) -> str:
        """Download media content and save it locally."""
        try:
            response = requests.get(media_url)
            if response.status_code == 200:
                # Determine file extension from content type
                content_type = response.headers.get('content-type', '').split('/')[-1]
                if content_type not in ['jpeg', 'jpg', 'png', 'gif']:
                    content_type = 'jpg'  # default to jpg if unknown
                
                file_path = os.path.join(self.media_dir, f"{tweet_id}.{content_type}")
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                return file_path
        except Exception as e:
            logging.error(f"Error downloading media for tweet {tweet_id}: {str(e)}")
        return None

    def process_tweet(self, tweet: tweepy.Status) -> Dict:
        """Process a single tweet and extract relevant information."""
        processed = {
            'tweet_id': tweet.id_str,
            'text': self.clean_text(tweet.text),
            'timestamp': tweet.created_at,
            'media_path': None
        }

        # Handle media content
        if hasattr(tweet, 'extended_entities') and 'media' in tweet.extended_entities:
            media = tweet.extended_entities['media'][0]
            if media['type'] in ['photo', 'animated_gif']:
                media_url = media['media_url_https']
                processed['media_path'] = self.download_media(media_url, tweet.id_str)

        return processed

    def fetch_tweets(self, query: str, count: int = 100) -> List[Dict]:
        """Fetch tweets based on a query and process them."""
        processed_tweets = []
        try:
            tweets = self.api.search_tweets(q=query, count=count, tweet_mode='extended')
            for tweet in tweets:
                processed = self.process_tweet(tweet)
                if processed['text'] and processed['media_path']:  # Only include tweets with both text and media
                    processed_tweets.append(processed)
        except Exception as e:
            logging.error(f"Error fetching tweets: {str(e)}")
        
        return processed_tweets

    def save_to_csv(self, tweets: List[Dict], output_file: str):
        """Save processed tweets to a CSV file."""
        df = pd.DataFrame(tweets)
        df.to_csv(output_file, index=False) 