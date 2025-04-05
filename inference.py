import torch
from PIL import Image
from torchvision import transforms
from model import MultimodalSentimentModel
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self, model_path: str = 'best_model.pth'):
        """Initialize the sentiment analyzer with a trained model."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = MultimodalSentimentModel()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess an image for model input."""
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image.unsqueeze(0)  # Add batch dimension
    
    def analyze_sentiment(self, text: str, image_path: str) -> dict:
        """Analyze sentiment of a tweet with its associated image."""
        try:
            # Preprocess image
            image = self.preprocess_image(image_path)
            image = image.to(self.device)
            
            # Make prediction
            with torch.no_grad():
                logits, intermediate_features = self.model(text, image)
                probabilities = torch.softmax(logits, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
            
            # Map prediction to sentiment
            sentiment_map = {0: 'sad', 1: 'neutral', 2: 'happy'}
            sentiment = sentiment_map[prediction]
            
            # Get confidence scores
            confidence = probabilities[0].cpu().numpy()
            
            return {
                'sentiment': sentiment,
                'confidence': {
                    'sad': float(confidence[0]),
                    'neutral': float(confidence[1]),
                    'happy': float(confidence[2])
                },
                'intermediate_features': {
                    'text_features': intermediate_features['text_features'].cpu().numpy(),
                    'image_features': intermediate_features['image_features'].cpu().numpy(),
                    'fused_features': intermediate_features['fused_features'].cpu().numpy()
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return {
                'error': str(e),
                'sentiment': None,
                'confidence': None
            }

def main():
    # Example usage
    analyzer = SentimentAnalyzer()
    
    # Example tweet and image
    text = "Just had the best day ever! #happy"
    image_path = "example_image.jpg"
    
    result = analyzer.analyze_sentiment(text, image_path)
    print(f"Sentiment: {result['sentiment']}")
    print(f"Confidence scores: {result['confidence']}")

if __name__ == '__main__':
    main() 