import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import timm
from typing import Tuple, Dict

class MultimodalSentimentModel(nn.Module):
    def __init__(self, num_classes: int = 3):
        """Initialize the multimodal sentiment analysis model."""
        super(MultimodalSentimentModel, self).__init__()
        
        # Text processing with BERT
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.text_projection = nn.Linear(768, 512)
        
        # Image processing with Swin Transformer
        self.swin = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        self.image_projection = nn.Linear(1024, 512)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )
        
        # Initialize tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
    def process_text(self, text: str) -> torch.Tensor:
        """Process text input through BERT."""
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self.bert(**inputs)
        text_features = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token representation
        return self.text_projection(text_features)
    
    def process_image(self, image: torch.Tensor) -> torch.Tensor:
        """Process image input through Swin Transformer."""
        image_features = self.swin.forward_features(image)
        image_features = image_features.mean(dim=1)  # Global average pooling
        return self.image_projection(image_features)
    
    def forward(self, text: str, image: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass through the model."""
        # Process text and image
        text_features = self.process_text(text)
        image_features = self.process_image(image)
        
        # Concatenate features
        combined_features = torch.cat([text_features, image_features], dim=1)
        
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
    
    def predict(self, text: str, image: torch.Tensor) -> str:
        """Make a prediction for a single input."""
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(text, image)
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            
        sentiment_map = {0: 'sad', 1: 'neutral', 2: 'happy'}
        return sentiment_map[prediction] 