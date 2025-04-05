import os
import pandas as pd
import numpy as np
from PIL import Image
import imageio
from typing import List, Dict, Tuple
import logging
from pathlib import Path
import re

class DataPreprocessor:
    def __init__(self, text_dir: str, image_dir: str, gif_dir: str, output_dir: str):
        """Initialize the DataPreprocessor with directory paths."""
        self.text_dir = Path(text_dir)
        self.image_dir = Path(image_dir)
        self.gif_dir = Path(gif_dir)
        self.output_dir = Path(output_dir)
        self.processed_images_dir = self.output_dir / "processed_images"
        
        # Create necessary directories
        self.processed_images_dir.mkdir(parents=True, exist_ok=True)
        
    def clean_text(self, text: str) -> str:
        """Clean text by removing special characters, URLs, etc."""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        # Remove emojis and special characters
        text = re.sub(r'[^\w\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def process_gif(self, gif_path: Path) -> List[Path]:
        """Process a GIF file into individual frames."""
        try:
            # Read GIF
            gif = imageio.mimread(gif_path)
            frame_paths = []
            
            # Save each frame
            for i, frame in enumerate(gif):
                frame_path = self.processed_images_dir / f"{gif_path.stem}_frame_{i}.png"
                Image.fromarray(frame).save(frame_path)
                frame_paths.append(frame_path)
            
            return frame_paths
        except Exception as e:
            logging.error(f"Error processing GIF {gif_path}: {str(e)}")
            return []
    
    def process_text_files(self) -> pd.DataFrame:
        """Process text files and create a DataFrame."""
        text_data = []
        
        for text_file in self.text_dir.glob("*.txt"):
            try:
                with open(text_file, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                    cleaned_text = self.clean_text(text)
                    
                    text_data.append({
                        'id': text_file.stem,
                        'text': cleaned_text,
                        'original_text': text
                    })
            except Exception as e:
                logging.error(f"Error processing text file {text_file}: {str(e)}")
        
        return pd.DataFrame(text_data)
    
    def process_image_files(self) -> Dict[str, Path]:
        """Process image files and return a mapping of IDs to image paths."""
        image_mapping = {}
        
        # Process regular images
        for image_file in self.image_dir.glob("*.{jpg,jpeg,png}"):
            try:
                # Copy image to processed directory
                target_path = self.processed_images_dir / image_file.name
                Image.open(image_file).save(target_path)
                image_mapping[image_file.stem] = target_path
            except Exception as e:
                logging.error(f"Error processing image {image_file}: {str(e)}")
        
        # Process GIFs
        for gif_file in self.gif_dir.glob("*.gif"):
            try:
                frame_paths = self.process_gif(gif_file)
                if frame_paths:
                    # Use the first frame as the representative image
                    image_mapping[gif_file.stem] = frame_paths[0]
            except Exception as e:
                logging.error(f"Error processing GIF {gif_file}: {str(e)}")
        
        return image_mapping
    
    def create_dataset(self) -> pd.DataFrame:
        """Create the final dataset by combining text and image data."""
        # Process text and images
        text_df = self.process_text_files()
        image_mapping = self.process_image_files()
        
        # Create dataset
        dataset = []
        for _, row in text_df.iterrows():
            text_id = row['id']
            if text_id in image_mapping:
                dataset.append({
                    'id': text_id,
                    'text': row['text'],
                    'original_text': row['original_text'],
                    'image_path': str(image_mapping[text_id])
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(dataset)
        
        # Save to CSV
        output_file = self.output_dir / "processed_dataset.csv"
        df.to_csv(output_file, index=False)
        
        return df
    
    def add_sentiment_labels(self, df: pd.DataFrame, label_file: str) -> pd.DataFrame:
        """Add sentiment labels to the dataset."""
        try:
            labels_df = pd.read_csv(label_file)
            df = df.merge(labels_df, on='id', how='left')
            return df
        except Exception as e:
            logging.error(f"Error adding sentiment labels: {str(e)}")
            return df 