# app.py - Amazon & Flipkart Review Sentiment Analyzer (BeautifulSoup version)
import streamlit as st
import requests
from bs4 import BeautifulSoup
import numpy as np
import pickle
import torch
import torch.nn as nn
import re
from typing import List, Tuple

# ───────────────────────────────────────────────────────────────
# DEVICE & MODEL CLASS DEFINITION (required for pickle load)
# ───────────────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes,
                 num_layers=1, dropout=0.3, pretrained_embeddings=None):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings).float())
        
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
        embed = self.embedding(x)
        lstm_out, (hn, cn) = self.lstm(embed)
        out = self.dropout(hn[-1])
        out = self.fc(out)
        return out

# ───────────────────────────────────────────────────────────────
# CONFIGURATION
# ───────────────────────────────────────────────────────────────
EMBEDDING_FILES = {
    'embeddings': 'ecommerce_word2vec_embeddings.npy',
    'word_to_id': 'ecommerce_word_to_id.npy',
    'id_to_word': 'ecommerce_id_to_word.npy',
}
MAX_LEN = 80
PAD_IDX = 0
UNK_IDX = 1
MODEL_PKL = 'lstm_model_full.pkl'  # Your pickled full model

# ───────────────────────────────────────────────────────────────
# Load embeddings & vocabulary
# ───────────────────────────────────────────────────────────────
@st.cache_data
def load_embeddings():
    try:
        embeddings = np.load(EMBEDDING_FILES['embeddings'])
        word_to_id = np.load(EMBEDDING_FILES['word_to_id'], allow_pickle=True).item()
        return embeddings, word_to_id
    except Exception as e:
        st.error(f"Failed to load embeddings: {e}")
        return None, None

# ───────────────────────────────────────────────────────────────
# Load trained LSTM model (full pickled object)
# ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_lstm_model():
    try:
        with open(MODEL_PKL, 'rb') as f:
            model = pickle.load(f)
        model.eval()
        model.to(DEVICE)
        return model
    except Exception as e:
        st.error(f"Failed to load LSTM model: {e}\n"
                 "Make sure SentimentLSTM class is defined above and file exists.")
        return None

# ───────────────────────────────────────────────────────────────
# Tokenization & Preparation
# ───────────────────────────────────────────────────────────────
def tokenize(text: str) -> List[str]:
    text = str(text).lower()
    pattern = re.compile(r"\b[a-zA-Z']+\b")
    return pattern.findall(text)

def text_to_sequence(text: str, word_to_id: dict, max_len: int = MAX_LEN) -> np.ndarray:
    tokens = tokenize(text)
    seq = [word_to_id.get(token, UNK_IDX) for token in tokens]
    seq = seq[:max_len]
    seq += [PAD_IDX] * (max_len - len(seq))
    return np.array(seq, dtype=np.int64)

# ───────────────────────────────────────────────────────────────
# Review Scraping with BeautifulSoup + requests
# ───────────────────────────────────────────────────────────────
def extract_reviews(url: str, max_pages: int = 3) -> List[str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    
    reviews = []
    current_url = url
    
    with st.spinner("Fetching reviews... (this may take a moment)"):
        for page in range(1, max_pages + 1):
            try:
                response = requests.get(current_url, headers=headers, timeout=30)
                if response.status_code != 200:
                    st.warning(f"Page {page} failed (status {response.status_code})")
                    break
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Amazon review extraction
                if "amazon" in url.lower():
                    review_elements = soup.select('span[data-hook="review-body"] span.a-size-base')
                    for el in review_elements:
                        txt = el.get_text(strip=True)
                        if txt and len(txt) > 20:
                            reviews.append(txt)
                
                # Flipkart review extraction
                elif "flipkart" in url.lower():
                    review_elements = soup.select('div.t-ZTKy span')
                    for el in review_elements:
                        txt = el.get_text(strip=True)
                        if txt and len(txt) > 20:
                            reviews.append(txt)
                
                # Next page (basic handling)
                if "amazon" in url.lower():
                    next_link = soup.select_one('li.a-last a')
                else:
                    next_link = soup.select_one('div._1LKTOy a, a._1LKTOy')
                
                if not next_link or not next_link.get('href'):
                    break
                
                current_url = next_link['href']
                if current_url.startswith('/'):
                    current_url = "https://www." + ("amazon.in" if "amazon" in url else "flipkart.com") + current_url
                
                time.sleep(4)  # Polite delay
                
            except Exception as e:
                st.error(f"Error on page {page}: {str(e)}")
                break
    
    unique_reviews = list(dict.fromkeys(reviews))
    st.info(f"Extracted {len(unique_reviews)} unique reviews")
    return unique_reviews[:60]

# ───────────────────────────────────────────────────────────────
# Sentiment Prediction
# ───────────────────────────────────────────────────────────────
def predict_sentiment(model, embeddings, seq: np.ndarray) -> float:
    if model is None or embeddings is None:
        return 0.5
    
    seq_tensor = torch.LongTensor(seq).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(seq_tensor)
        probs = torch.softmax(output, dim=1)
        score = probs[0, 0].item()  # Assuming class 0 = Positive
    
    return score

def get_sentiment_label(score: float) -> Tuple[str, str]:
    if score < 0.35:
        return "Negative 😞", "red"
    elif score < 0.65:
        return "Neutral 😐", "orange"
    else:
        return "Positive 😊", "green"

# ───────────────────────────────────────────────────────────────
# STREAMLIT APP
# ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Product Review Sentiment Analyzer", layout="wide")

st.title("🛍️ Amazon & Flipkart Review Sentiment Analyzer")
st.markdown("Analyze customer sentiment using your custom LSTM model + Word2Vec embeddings.")

# Load resources
embeddings, word_to_id = load_embeddings()
model = load_lstm_model()

if embeddings is None or word_to_id is None or model is None:
    st.stop()

product_url = st.text_input(
    "Product URL (Amazon or Flipkart)",
    placeholder="https://www.amazon.in/... or https://www.flipkart.com/..."
)

col1, col2 = st.columns(2)
with col1:
    max_pages = st.slider("Max pages to scrape", 1, 5, 3)

if st.button("Scrape & Analyze", type="primary"):
    if not product_url:
        st.error("Please enter a product URL")
    elif "amazon" not in product_url.lower() and "flipkart" not in product_url.lower():
        st.error("Only Amazon & Flipkart links supported")
    else:
        reviews = extract_reviews(product_url, max_pages)
        
        if reviews:
            st.success(f"Extracted {len(reviews)} reviews! Analyzing sentiment...")
            
            progress = st.progress(0)
            results = []
            
            for i, review in enumerate(reviews):
                seq = text_to_sequence(review, word_to_id)
                score = predict_sentiment(model, embeddings, seq)
                label, color = get_sentiment_label(score)
                results.append((review[:250] + ("..." if len(review)>250 else ""), score, label, color))
                progress.progress((i + 1) / len(reviews))
            
            # Overall result
            avg_score = np.mean([r[1] for r in results])
            avg_label, avg_color = get_sentiment_label(avg_score)
            
            st.markdown(f"""
            <div style="text-align:center; padding:20px; background:{avg_color}22; border-radius:10px;">
                <h2 style="color:{avg_color};">Overall Sentiment: {avg_label}</h2>
                <h3>Average Score: {avg_score:.3f}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Individual reviews
            st.markdown("### Review Breakdown")
            for review_text, score, label, color in results:
                st.markdown(f"""
                <div style="border-left:4px solid {color}; padding-left:15px; margin:10px 0;">
                    <strong>{label}</strong> (Score: {score:.3f})<br>
                    {review_text}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No reviews found. Try a different product or check the link.")

st.info("Note: Scraping uses polite delays. Sites may limit access or change layout.")