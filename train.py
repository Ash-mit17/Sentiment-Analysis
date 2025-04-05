import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from model import MultimodalSentimentModel
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TweetDataset(Dataset):
    def __init__(self, data: pd.DataFrame, transform=None):
        self.data = data
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = row['text']
        image = Image.open(row['media_path']).convert('RGB')
        image = self.transform(image)
        label = row['sentiment']
        return text, image, label

def train_model(model: nn.Module, 
                train_loader: DataLoader, 
                val_loader: DataLoader,
                num_epochs: int = 10,
                learning_rate: float = 1e-4,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """Train the multimodal sentiment analysis model."""
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_accuracy = 0
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for texts, images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(texts, images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for texts, images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                logits, _ = model(texts, images)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        val_accuracy = accuracy_score(all_labels, all_preds)
        val_accuracies.append(val_accuracy)
        
        logger.info(f'Epoch {epoch+1}/{num_epochs}:')
        logger.info(f'Train Loss: {avg_train_loss:.4f}')
        logger.info(f'Val Loss: {avg_val_loss:.4f}')
        logger.info(f'Val Accuracy: {val_accuracy:.4f}')
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.savefig('training_curves.png')
    plt.close()
    
    return model

def evaluate_model(model: nn.Module, test_loader: DataLoader, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """Evaluate the model on the test set."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for texts, images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            logits, _ = model(texts, images)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)
    
    logger.info(f'Test Accuracy: {accuracy:.4f}')
    logger.info(f'Test F1 Score: {f1:.4f}')
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    return accuracy, f1, cm

if __name__ == '__main__':
    # Load and preprocess data
    data = pd.read_csv('processed_tweets.csv')
    
    # Split data
    train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    # Create datasets
    train_dataset = TweetDataset(train_data)
    val_dataset = TweetDataset(val_data)
    test_dataset = TweetDataset(test_data)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Initialize model
    model = MultimodalSentimentModel()
    
    # Train model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trained_model = train_model(model, train_loader, val_loader, device=device)
    
    # Evaluate model
    evaluate_model(trained_model, test_loader, device=device) 