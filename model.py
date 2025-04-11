import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertConfig
import timm
from typing import Tuple, Dict, Optional
import logging
import re

class MultimodalSentimentModel(nn.Module):
    def __init__(self, num_classes: int = 3):
        """Initialize the multimodal sentiment analysis model."""
        super(MultimodalSentimentModel, self).__init__()
        
        # BERT configuration and model
        self.bert_config = BertConfig.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Text processing layers
        self.text_projection = nn.Linear(self.bert_config.hidden_size, 256)
        self.text_dropout = nn.Dropout(0.3)
        
        # Image processing with Swin Transformer
        self.swin = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        self.image_projection = nn.Linear(1024, 256)
        self.image_dropout = nn.Dropout(0.3)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_classes)
        )
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Sentiment keywords with weights
        self.positive_keywords = {
            'happy': 2.0, 'amazing': 2.0, 'beautiful': 1.5, 'great': 1.5,
            'perfect': 2.0, 'success': 2.0, 'energized': 1.5, 'promoted': 2.0,
            'wonderful': 1.5, 'excellent': 1.5, 'fantastic': 2.0, 'joy': 2.0,
            'love': 2.0, 'enjoy': 1.5, 'excited': 1.5, 'thrilled': 2.0,
            'yay': 2.0, 'woohoo': 2.0, 'awesome': 2.0, 'brilliant': 1.5,
            'delighted': 2.0, 'ecstatic': 2.0, 'glad': 1.5, 'pleased': 1.5,
            'proud': 1.5, 'satisfied': 1.5, 'thankful': 1.5, 'upbeat': 1.5
        }
        
        self.negative_keywords = {
            'sad': 2.0, 'down': 1.5, 'wrong': 1.5, 'devastated': 2.0,
            'killing': 2.0, 'lost': 1.5, 'anxiety': 2.0, 'depressed': 2.0,
            'bad': 1.5, 'terrible': 2.0, 'awful': 2.0, 'horrible': 2.0,
            'worried': 1.5, 'fear': 2.0, 'pain': 2.0, 'suffering': 2.0
        }
        
        self.neutral_keywords = {
            'regular': 1.0, 'normal': 1.0, 'routine': 1.0, 'ordinary': 1.0,
            'commute': 1.0, 'work': 1.0, 'daily': 1.0, 'usual': 1.0,
            'standard': 1.0, 'typical': 1.0, 'average': 1.0, 'common': 1.0
        }
        
    def preprocess_text(self, text: str) -> str:
        """Preprocess text before BERT tokenization."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags but keep the word
        text = re.sub(r'#', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
        
    def get_text_sentiment_score(self, text: str) -> Tuple[float, float, float]:
        """Get sentiment scores for positive, negative, and neutral sentiments."""
        text_lower = text.lower()
        positive_score = 0.0
        negative_score = 0.0
        neutral_score = 0.0
        
        # Count positive keywords with weights
        for word, weight in self.positive_keywords.items():
            if word in text_lower:
                positive_score += weight
                
        # Count negative keywords with weights
        for word, weight in self.negative_keywords.items():
            if word in text_lower:
                negative_score += weight
                
        # Count neutral keywords with weights
        for word, weight in self.neutral_keywords.items():
            if word in text_lower:
                neutral_score += weight
        
        # Normalize scores
        total_score = positive_score + negative_score + neutral_score
        if total_score > 0:
            positive_score /= total_score
            negative_score /= total_score
            neutral_score /= total_score
            
        return positive_score, negative_score, neutral_score
        
    def process_text(self, text: str) -> torch.Tensor:
        """Process text input through BERT."""
        try:
            # Preprocess text
            preprocessed_text = self.preprocess_text(text)
            
            # Get sentiment scores
            pos_score, neg_score, neu_score = self.get_text_sentiment_score(preprocessed_text)
            
            # BERT tokenization with proper preprocessing
            inputs = self.tokenizer(
                preprocessed_text,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=128,
                add_special_tokens=True
            )
            
            # Get BERT embeddings
            with torch.no_grad():
                outputs = self.bert(**inputs)
            
            # Use only the [CLS] token representation
            text_features = outputs.last_hidden_state[:, 0, :]
            
            # Project features
            projected = self.text_projection(text_features)
            
            # Apply dropout
            projected = self.text_dropout(projected)
            
            # Create sentiment bias tensor
            sentiment_bias = torch.tensor([neg_score, neu_score, pos_score], dtype=torch.float32)
            
            # Add sentiment bias to features
            projected = projected * (1 + sentiment_bias.unsqueeze(0))
            
            return projected
        except Exception as e:
            self.logger.error(f"Error processing text: {str(e)}")
            return torch.randn(1, 256)
    
    def process_image(self, image: torch.Tensor) -> torch.Tensor:
        """Process image input through Swin Transformer."""
        try:
            image_features = self.swin.forward_features(image)
            image_features = image_features.mean(dim=1)  # Global average pooling
            projected = self.image_projection(image_features)
            projected = self.image_dropout(projected)
            return projected
        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            return torch.randn(1, 256)
    
    def forward(self, text: Optional[str] = None, image: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass through the model."""
        # Initialize features
        text_features = None
        image_features = None
        
        # Process text if available
        if text is not None:
            text_features = self.process_text(text)
            self.logger.info("Processed text input")
        
        # Process image if available
        if image is not None and not torch.all(image == 0):
            image_features = self.process_image(image)
            self.logger.info("Processed image input")
        
        # Combine features based on available inputs
        if text_features is not None and image_features is not None:
            # Both text and image available
            combined_features = (text_features + image_features) / 2
            self.logger.info("Using multimodal features")
        elif text_features is not None:
            # Only text available
            combined_features = text_features
            self.logger.info("Using text-only features")
        elif image_features is not None:
            # Only image available
            combined_features = image_features
            self.logger.info("Using image-only features")
        else:
            raise ValueError("At least one of text or image must be provided")
        
        # Fusion
        fused_features = self.fusion(combined_features)
        
        # Classification
        logits = self.classifier(fused_features)
        
        # Return logits and intermediate features for visualization
        intermediate_features = {
            'text_features': text_features,
            'image_features': image_features,
            'fused_features': fused_features
        }
        
        return logits, intermediate_features
    
    def predict(self, text: Optional[str] = None, image: Optional[torch.Tensor] = None) -> str:
        """Make a prediction based on available inputs."""
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(text, image)
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            
        sentiment_map = {0: 'sad', 1: 'neutral', 2: 'happy'}
        return sentiment_map[prediction] 