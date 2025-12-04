import os
import json
import torch
import pickle
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Disable TensorFlow warnings and set environment variables
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

# Load JSON dataset
with open('dataset.json', 'r') as f:
    dataset = json.load(f)

# Extract prompts and labels
prompts = [item['prompt'] for item in dataset]
labels = [item['label'] for item in dataset]

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
num_labels = len(set(labels))  # Should be 5

# Simple text preprocessing function
def build_vocab(prompts, vocab_size=1000):
    """Build vocabulary from prompts"""
    vocab = {}
    for prompt in prompts:
        for word in prompt.lower().split():
            if word not in vocab and len(vocab) < vocab_size:
                vocab[word] = len(vocab) + 1  # +1 to reserve 0 for padding
    return vocab

def simple_tokenize(text, vocab, max_length=20):
    """Convert text to numbers using provided vocab"""
    words = text.lower().split()
    token_ids = [vocab.get(word, 0) for word in words]
    if len(token_ids) > max_length:
        token_ids = token_ids[:max_length]
    else:
        token_ids.extend([0] * (max_length - len(token_ids)))
    return torch.tensor(token_ids, dtype=torch.long)

# Build vocabulary
vocab = build_vocab(prompts)

# Custom Dataset class
class PromptDataset(Dataset):
    def __init__(self, prompts, labels, vocab):
        self.prompts = prompts
        self.labels = labels
        self.vocab = vocab
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        label = self.labels[idx]
        tokenized = simple_tokenize(prompt, self.vocab)
        return tokenized, label

# Simple neural network for text classification
class SimpleTextClassifier(nn.Module):
    def __init__(self, vocab_size=1000, embedding_dim=50, hidden_dim=100, num_classes=5):
        super(SimpleTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        output = self.fc(self.dropout(hidden[-1]))
        return output

# Create model
model = SimpleTextClassifier(vocab_size=1000, num_classes=num_labels)

# Prepare data
dataset = PromptDataset(prompts, encoded_labels, vocab)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Train the model
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()
for epoch in range(50):
    total_loss = 0
    for batch_prompts, batch_labels in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_prompts)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

# Create neural_network_model folder if it doesn't exist
model_folder = Path("neural_network_model")
model_folder.mkdir(exist_ok=True)

# Save the model, label encoder, and vocabulary in neural_network_model folder
model_path = model_folder / 'model.pth'
label_encoder_path = model_folder / 'label_encoder.pkl'
vocab_path = model_folder / 'vocab.pkl'

torch.save(model.state_dict(), model_path)
with open(label_encoder_path, 'wb') as f:
    pickle.dump(label_encoder, f)
with open(vocab_path, 'wb') as f:
    pickle.dump(vocab, f)

print(f"Model, label encoder, and vocabulary saved successfully in '{model_folder}' folder.")
print(f"  - Model: {model_path}")
print(f"  - Label Encoder: {label_encoder_path}")
print(f"  - Vocabulary: {vocab_path}")

# Function to predict model type (for testing)
def predict_model_type(prompt, model, vocab, label_encoder):
    model.eval()
    with torch.no_grad():
        tokenized = simple_tokenize(prompt, vocab).unsqueeze(0)
        output = model(tokenized)
        predicted_class = torch.argmax(output, dim=1).item()
        predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    return predicted_label

# Example usage
new_prompt = "help me predict tsunami"
predicted_model = predict_model_type(new_prompt, model, vocab, label_encoder)
print(f"Recommended model for '{new_prompt}': {predicted_model}")