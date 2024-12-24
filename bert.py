import pandas as pd
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch
from transformers import AdamW
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
# Load dataset
df = pd.read_csv('sentiment140.csv', encoding='latin1', header=None)
df.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']

# Map sentiment to binary values (0 = negative, 1 = positive)
df['sentiment'] = df['sentiment'].map({0: 0, 4: 1})

# Focus on text and sentiment
df = df[['text', 'sentiment']]
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
try:
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
def preprocess(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)    # Remove mentions
    text = re.sub(r'#\w+', '', text)    # Remove hashtags
    text = re.sub(r'\d+', '', text)     # Remove numbers
    text = re.sub(r'[^\w\s]', '', text) # Remove special characters
    text = text.lower()                 # Lowercase
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Apply preprocessing
df['cleaned_text'] = df['text'].apply(preprocess)
tfidf = TfidfVectorizer(max_features=1000, max_df=0.8, min_df=5)
X = tfidf.fit_transform(df['cleaned_text'])
y = df['sentiment'].values
print("hey1")
svd = TruncatedSVD(n_components=100, random_state=42)
X_reduced = svd.fit_transform(X)
print("hey2")
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)
print("hey3")
# Train a Random Forest
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
print("hey4")
# Evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
# Load tokenizer and model
print("hey5")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class SentimentDataset(Dataset):
    def _init_(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def _len_(self):
        return len(self.texts)

    def _getitem_(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = tokenizer.encode_plus(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {'input_ids': encoding['input_ids'].squeeze(), 
                'attention_mask': encoding['attention_mask'].squeeze(),
                'labels': torch.tensor(label, dtype=torch.long)}

# Prepare datasets
train_dataset = SentimentDataset(df['cleaned_text'].tolist(), df['sentiment'].tolist())
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
print("hey6")
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
optimizer = AdamW(model.parameters(), lr=1e-5)
print("hey7")
# Training loop
epochs = 3
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} loss: {loss.item()}")

def evaluate_model(loader, model):
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, axis=1)
            predictions.extend(preds)
            true_labels.extend(labels)

    return accuracy_score(true_labels, predictions)

accuracy = evaluate_model(train_loader, model)
print("BERT Model Accuracy:", accuracy)

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)