# Multimodal Twitter Sentiment Analysis

This project implements a multimodal sentiment analysis model that combines text and image features to classify tweets into happy, sad, or neutral sentiments.

## Project Structure

- `data_preprocessor.py`: Handles local data preprocessing for text, images, and GIFs
- `model.py`: Contains the multimodal sentiment analysis model architecture
- `train.py`: Training script for the model
- `inference.py`: Script for making predictions with the trained model
- `requirements.txt`: Project dependencies

## Data Organization

Organize your data in the following structure:
```
data/
├── text/
│   ├── tweet_1.txt
│   ├── tweet_2.txt
│   └── ...
├── images/
│   ├── tweet_1.jpg
│   ├── tweet_2.png
│   └── ...
├── gifs/
│   ├── tweet_3.gif
│   ├── tweet_4.gif
│   └── ...
└── labels.csv
```

The `labels.csv` file should contain:
- `id`: Matching the filename stem (e.g., "tweet_1" for "tweet_1.txt")
- `sentiment`: 0 (sad), 1 (neutral), or 2 (happy)

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Organize your data files as described above

## Data Preprocessing

The preprocessing pipeline handles:
- Text cleaning (removing URLs, mentions, hashtags, special characters)
- Image processing (resizing, normalization)
- GIF processing (extracting frames, using first frame as representative)
- Data combination and label integration

To preprocess your data:
```python
from data_preprocessor import DataPreprocessor

preprocessor = DataPreprocessor(
    text_dir="data/text",
    image_dir="data/images",
    gif_dir="data/gifs",
    output_dir="processed_data"
)

# Create the dataset
dataset = preprocessor.create_dataset()

# Add sentiment labels
dataset = preprocessor.add_sentiment_labels(dataset, "data/labels.csv")
```

## Model Architecture

The model combines:
1. Text Processing:
   - BERT for text feature extraction
   - Output: 768-dimensional text embeddings

2. Image Processing:
   - Swin Transformer for image feature extraction
   - Handles both static images and GIF frames
   - Output: 1024-dimensional image embeddings

3. Feature Fusion:
   - Concatenates text and image embeddings
   - Multi-layer perceptron for feature fusion
   - Dropout for regularization

4. Classification:
   - Softmax layer for sentiment prediction
   - Three classes: happy, sad, neutral

## Training Process

1. Data Loading:
   - Loads preprocessed dataset
   - Splits into train/val/test sets (70/15/15)
   - Creates PyTorch DataLoaders

2. Model Training:
   - Uses Adam optimizer
   - Cross-entropy loss
   - Early stopping based on validation loss
   - Saves best model weights

3. Evaluation:
   - Accuracy and F1-score metrics
   - Confusion matrix visualization
   - Training curves plotting

To train the model:
```bash
python train.py
```

## Inference

To analyze sentiment of new content:
```python
from inference import SentimentAnalyzer

analyzer = SentimentAnalyzer()
result = analyzer.analyze_sentiment(
    text="Your text here",
    image_path="path_to_image.jpg"
)

print(f"Sentiment: {result['sentiment']}")
print(f"Confidence scores: {result['confidence']}")
```

## Results

The model outputs:
1. Predicted sentiment (happy/sad/neutral)
2. Confidence scores for each class
3. Intermediate features for visualization

## Performance Metrics

The model is evaluated using:
- Accuracy: Overall prediction correctness
- F1-score: Balance between precision and recall
- Confusion Matrix: Detailed class-wise performance

## Visualization

The training process generates:
1. Training curves:
   - Training and validation loss
   - Validation accuracy over epochs

2. Confusion matrix:
   - Visual representation of predictions
   - Class-wise performance analysis

## License

This project is licensed under the MIT License - see the LICENSE file for details. 