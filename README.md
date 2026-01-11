# 🛍️ E-Commerce Review Sentiment Analysis using Custom Word2Vec + LSTM

This project performs **sentiment analysis on e-commerce product reviews** using a **Word2Vec model built completely from scratch with NumPy** and a **custom LSTM classifier implemented in PyTorch**.

The project emphasizes **understanding core NLP and deep learning concepts** rather than relying on high-level pretrained models.

---

## 📌 Project Overview

The system is divided into three major stages:

1. **Word2Vec embedding training from scratch**
2. **LSTM-based sentiment classification**
3. **Inference using a Streamlit web application**

Sentiments are classified into:
- **Positive**
- **Neutral**
- **Negative**

---

## 🧠 Word2Vec Implementation (From Scratch)

This project implements **Skip-Gram Word2Vec** without using libraries like Gensim.

### Key Features
- One-hot encoded input vectors
- Sliding context window
- Manual softmax computation
- Cross-entropy loss
- Gradient descent optimization
- Implemented fully using NumPy

### Training Configuration
- Context window size: 5
- Embedding dimension: 30
- Learning rate: 0.0005
- Iterations: 2500

### Generated Outputs
- `ecommerce_word2vec_embeddings.npy`
- `ecommerce_word_to_id.npy`
- `ecommerce_id_to_word.npy`
- `ecommerce_word2vec.txt`

These embeddings are later **directly loaded into the LSTM embedding layer**.

---

## 🔍 Text Preprocessing

- Lowercasing
- Regex-based tokenization
- Vocabulary mapping
- `<pad>` and `<unk>` tokens
- Fixed-length padding/truncation (MAX_LEN = 80)

This ensures consistent preprocessing during both training and inference.

---

## 🔁 LSTM Sentiment Classifier

### Model Architecture
- Embedding Layer (initialized with custom Word2Vec weights)
- Single-layer unidirectional LSTM
- Hidden size: 128
- Dropout: 0.3
- Fully connected output layer
- Softmax activation

### Label Encoding
| Sentiment | Class ID |
|---------|----------|
| Positive | 0 |
| Neutral | 1 |
| Negative | 2 |

---

## ⚙️ Training Configuration

| Parameter | Value |
|---------|------|
| Batch size | 64 |
| Epochs | 60 |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Loss function | CrossEntropyLoss |
| Gradient clipping | Enabled (max norm = 1.0) |
| Device | CPU / CUDA (if available) |

---

## 📊 Evaluation Metrics Used

The model is evaluated using **multiple standard classification metrics**.

### 1️⃣ Accuracy
- Used during **training**, **validation**, and **testing**
- Best model selected based on **validation accuracy**

Test Accuracy ≈ 75% – 83%

*(Exact value may vary slightly due to random initialization and data split)*

---

### 2️⃣ Cross-Entropy Loss
- Used as the **training objective**
- Monitored for:
  - Training loss
  - Validation loss
- Helps measure confidence of predictions

---

### 3️⃣ Precision, Recall, and F1-Score
A **classification report** is generated on the test set:

- Precision: Correctness of predicted labels
- Recall: Coverage of actual labels
- F1-score: Balance between precision and recall

Reported separately for:
- Positive
- Neutral
- Negative

---

### 4️⃣ Confusion Matrix
- Visualized using a heatmap
- Helps analyze:
  - Class-wise misclassification
  - Neutral vs Positive confusion
  - Negative prediction errors

---

## 🧪 Model Selection Strategy

- Dataset split:
  - Training
  - Validation (15%)
  - Testing
- Best model checkpoint selected using **highest validation accuracy**
- Final evaluation performed on the unseen test set

---

## 💾 Model Saving Strategy

The **entire trained LSTM model object** is saved using Python `pickle`:

lstm_model_full.pkl

This allows:
- Direct inference without redefining architecture
- Seamless integration with Streamlit

---

## 🖥️ Streamlit Application

### Features
- Load trained Word2Vec embeddings
- Load trained LSTM model
- Convert input text to sequences
- Predict sentiment with confidence score
- Display:
  - Individual review sentiment
  - Overall average sentiment

### Scraping Disclaimer
- Uses BeautifulSoup + requests
- Subject to website layout changes and restrictions
- **Not claimed as real-time or fully reliable**

---

## Limitations

- Scraping may fail due to anti-bot protection
- Dataset-specific vocabulary

---

## 💡 Future Work (Concept Only)

- Replace scraping with official APIs
- Online learning / incremental updates
- Aspect-based sentiment analysis
- Attention-enhanced LSTM
- Real-time dashboards

---

## 🧰 Tech Stack

- Python
- NumPy
- PyTorch
- Streamlit
- BeautifulSoup
- scikit-learn
- Matplotlib


## 🎯 Learning Outcomes

- Implemented Word2Vec from scratch
- Built custom NLP pipelines
- Trained and evaluated LSTM models
- Applied evaluation metrics correctly
- Deployed ML models using Streamlit

---

Clone this repository:  
   ```bash
   git clone https://github.com/Dhiyanesh-B/Sentiment-Analysis.git
```
Or download the ZIP and extract it.

Install the required libraries using this command
   ```bash
    pip install numpy pandas torch streamlit scikit-learn matplotlib seaborn beautifulsoup4 requests tqdm
```
---

## Team Members
1. Dhiyanesh B
2. [Sabariesh Karthic A M](https://github.com/sabarieshkarthic)

---
⭐ **If you like this project, give it a star!** ⭐
