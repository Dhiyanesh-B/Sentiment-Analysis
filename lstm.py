import numpy as np
import pandas as pd
import re
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ───────────────────────────────────────────────────────────────
# CONFIGURATION
# ───────────────────────────────────────────────────────────────
EMBEDDING_FILES = {
    'embeddings': 'ecommerce_word2vec_embeddings.npy',
    'word_to_id': 'ecommerce_word_to_id.npy',
    'id_to_word': 'ecommerce_id_to_word.npy',
}

MAX_LEN = 80
EMBEDDING_DIM = 30
HIDDEN_SIZE = 128
NUM_LAYERS = 1
DROPOUT = 0.3
BATCH_SIZE = 64
EPOCHS = 60
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LABEL_MAP = {
    'Positive': 0,
    'Neutral':  1,
    'Negative': 2
}

# Special token indices
PAD_IDX = 0
UNK_IDX = 1

# ───────────────────────────────────────────────────────────────
# 1. Load & Prepare Vocabulary + Embeddings with special tokens
# ───────────────────────────────────────────────────────────────
print("Loading pre-trained embeddings...")
orig_embeddings = np.load(EMBEDDING_FILES['embeddings'])
orig_word_to_id = np.load(EMBEDDING_FILES['word_to_id'], allow_pickle=True).item()
orig_id_to_word = np.load(EMBEDDING_FILES['id_to_word'], allow_pickle=True).item()

ORIGINAL_VOCAB_SIZE = orig_embeddings.shape[0]

# Create new vocabulary with PAD and UNK
word_to_id = {'<pad>': PAD_IDX, '<unk>': UNK_IDX}
id_to_word = {PAD_IDX: '<pad>', UNK_IDX: '<unk>'}

next_id = 2
for word, old_id in sorted(orig_word_to_id.items(), key=lambda x: x[1]):
    word_to_id[word] = next_id
    id_to_word[next_id] = word
    next_id += 1

VOCAB_SIZE = len(word_to_id)  # 1134 + 2 = 1136

# New embedding matrix
embeddings = np.zeros((VOCAB_SIZE, EMBEDDING_DIM), dtype=np.float32)
# <unk> = average of known embeddings
embeddings[UNK_IDX] = np.mean(orig_embeddings, axis=0)
# Copy original embeddings (shifted by +2)
for word, new_id in word_to_id.items():
    if word in orig_word_to_id:
        embeddings[new_id] = orig_embeddings[orig_word_to_id[word]]

print(f"Final vocabulary size (with <pad>/<unk>): {VOCAB_SIZE}")
print(f"Embedding dimension: {EMBEDDING_DIM}")

# ───────────────────────────────────────────────────────────────
# 2. Tokenization (same as your original)
# ───────────────────────────────────────────────────────────────
def tokenize(text):
    text = str(text).lower()
    pattern = re.compile(r"\b[a-zA-Z']+\b")
    return pattern.findall(text)

# ───────────────────────────────────────────────────────────────
# 3. Text → sequence (now using new vocabulary!)
# ───────────────────────────────────────────────────────────────
def text_to_sequence(text, max_len=MAX_LEN):
    tokens = tokenize(text)
    seq = [word_to_id.get(token, UNK_IDX) for token in tokens]
    
    # Truncate
    seq = seq[:max_len]
    # Pad
    seq = seq + [PAD_IDX] * (max_len - len(seq))
    
    return np.array(seq, dtype=np.int64)

# ───────────────────────────────────────────────────────────────
# 4. Dataset
# ───────────────────────────────────────────────────────────────
class EcommerceSentimentDataset(Dataset):
    def __init__(self, texts, labels, max_len=MAX_LEN):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        seq = text_to_sequence(text, self.max_len)
        return {
            'input_ids': torch.LongTensor(seq),
            'label': torch.tensor(label, dtype=torch.long)
        }

# ───────────────────────────────────────────────────────────────
# 5. LSTM Model
# ───────────────────────────────────────────────────────────────
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes,
                 num_layers=1, dropout=0.3, pretrained_embeddings=None):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings).float())
            # self.embedding.weight.requires_grad = False  # uncomment to freeze
        
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        embed = self.embedding(x)                    # (batch, seq, embed_dim)
        lstm_out, (hn, cn) = self.lstm(embed)        # hn: (num_layers, batch, hidden)
        out = self.dropout(hn[-1])                   # last layer hidden
        out = self.fc(out)                           # (batch, num_classes)
        return out

# ───────────────────────────────────────────────────────────────
# 6. Training / Evaluation
# ───────────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, pred = outputs.max(1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(loader), correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    preds, trues = [], []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, pred = outputs.max(1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
            
            preds.extend(pred.cpu().numpy())
            trues.extend(labels.cpu().numpy())
    
    return total_loss / len(loader), correct / total, preds, trues

# ───────────────────────────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    df_train = pd.read_excel("train.xlsx")
    df_test  = pd.read_excel("test.xlsx")
    
    TEXT_COL = "reviews.text"
    LABEL_COL = "sentiment"
    
    # Map labels
    df_train['label_id'] = df_train[LABEL_COL].map(LABEL_MAP)
    df_test['label_id']  = df_test[LABEL_COL].map(LABEL_MAP)
    
    df_train = df_train.dropna(subset=['label_id'])
    df_test  = df_test.dropna(subset=['label_id'])
    
    print("Train label distribution:")
    print(df_train[LABEL_COL].value_counts())
    print(f"Total training samples after cleaning: {len(df_train)}")
    
    # Split train → train/val
    train_df, val_df = train_test_split(
        df_train, test_size=0.15, stratify=df_train['label_id'], random_state=42
    )
    
    # Datasets & Loaders
    ds_train = EcommerceSentimentDataset(train_df[TEXT_COL].values, train_df['label_id'].values.astype(int))
    ds_val   = EcommerceSentimentDataset(val_df[TEXT_COL].values,   val_df['label_id'].values.astype(int))
    ds_test  = EcommerceSentimentDataset(df_test[TEXT_COL].values,  df_test['label_id'].values.astype(int))
    
    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    dl_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE*2, shuffle=False, num_workers=0)
    dl_test  = DataLoader(ds_test,  batch_size=BATCH_SIZE*2, shuffle=False, num_workers=0)
    
    # Model
    model = SentimentLSTM(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBEDDING_DIM,
        hidden_size=HIDDEN_SIZE,
        num_classes=len(LABEL_MAP),
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        pretrained_embeddings=embeddings
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    best_acc = 0.0
    print("\nStarting training...\n")
    
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, dl_train, criterion, optimizer, DEVICE)
        val_loss, val_acc, _, _ = evaluate(model, dl_val, criterion, DEVICE)
        
        print(f"Epoch {epoch:2d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val   Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        scheduler.step(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_sentiment_lstm.pt")
            print("  → Saved new best model")
    
    # Final test evaluation
    print("\n" + "═"*65)
    print("Final evaluation on TEST set")
    model.load_state_dict(torch.load("best_sentiment_lstm.pt"))
    _, test_acc, preds, trues = evaluate(model, dl_test, criterion, DEVICE)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(trues, preds, target_names=['Positive', 'Neutral', 'Negative'], digits=3))
    
    # Confusion Matrix
    cm = confusion_matrix(trues, preds)
    plt.figure(figsize=(7,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Positive','Neutral','Negative'],
                yticklabels=['Positive','Neutral','Negative'])
    plt.title("Confusion Matrix - Test Set")
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()
    # Add this after your training loop, when model is trained
    import pickle

# Save the entire model object (not just state_dict)
    with open('lstm_model_full.pkl', 'wb') as f:
        pickle.dump(model, f)  # model is your SentimentLSTM instance

    print("LSTM model saved as 'lstm_model_full.pkl' - ready for Streamlit!")

if __name__ == "__main__":
    main()