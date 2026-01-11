import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

np.random.seed(42)

# --------------------- Softmax & Loss ---------------------
def softmax(X):
    res = []
    for x in X:
        exp = np.exp(x - np.max(x))
        res.append(exp / exp.sum())
    return np.array(res)

def cross_entropy(z, y):
    return -np.sum(np.log(z + 1e-10) * y)

# --------------------- Tokenization (Skip Labels) ---------------------
def tokenize(text):
    # Remove labels like __label__1, __label__2, etc.
    #text = re.sub(r'__label__\d+\s*', ' ', text)
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(text.lower())

# --------------------- Vocabulary & Data Generation ---------------------
def mapping(tokens):
    word_to_id = {}
    id_to_word = {}
    for i, token in enumerate(sorted(set(tokens))):
        word_to_id[token] = i
        id_to_word[i] = token
    return word_to_id, id_to_word

def concat(*iterables):
    for iterable in iterables:
        yield from iterable

def one_hot_encode(idx, vocab_size):
    res = [0] * vocab_size
    res[idx] = 1
    return res

def generate_training_data(tokens, word_to_id, window=2):
    X, y = [], []
    n_tokens = len(tokens)
    vocab_size = len(word_to_id)
    
    for i in range(n_tokens):
        context_indices = list(concat(
            range(max(0, i - window), i),
            range(i + 1, min(n_tokens, i + window + 1))
        ))
        for j in context_indices:
            X.append(one_hot_encode(word_to_id[tokens[i]], vocab_size))
            y.append(one_hot_encode(word_to_id[tokens[j]], vocab_size))
    
    return np.array(X), np.array(y)

# --------------------- Model Initialization ---------------------
def init_network(vocab_size, n_embedding):
    return {
        "w1": np.random.randn(vocab_size, n_embedding) * 0.01,
        "w2": np.random.randn(n_embedding, vocab_size) * 0.01
    }

# --------------------- Forward & Backward ---------------------
def forward(model, X, return_cache=True):
    cache = {}
    cache["a1"] = X @ model["w1"]
    cache["a2"] = cache["a1"] @ model["w2"]
    cache["z"] = softmax(cache["a2"])
    if not return_cache:
        return cache["z"]
    return cache

def backward(model, X, y, alpha):
    cache = forward(model, X)
    da2 = cache["z"] - y
    dw2 = cache["a1"].T @ da2
    da1 = da2 @ model["w2"].T
    dw1 = X.T @ da1
    
    model["w1"] -= alpha * dw1
    model["w2"] -= alpha * dw2
    
    return cross_entropy(cache["z"], y)

# --------------------- Load Data from train.txt ---------------------
def load_reviews_from_file(filename='train.txt'):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"{filename} not found in current directory!")
    
    all_tokens = []
    print(f"Loading reviews from {filename}...")
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            # Tokenize and skip labels automatically via tokenize()
            tokens = tokenize(line)
            all_tokens.extend(tokens)
            
            if line_num % 1000 == 0:
                print(f"Processed {line_num} lines...")
    
    print(f"Total tokens collected: {len(all_tokens)}")
    print(f"Unique words (vocab size): {len(set(all_tokens))}")
    return all_tokens

# --------------------- Main Training ---------------------
# Load text from train.txt (labels are automatically ignored)
tokens = load_reviews_from_file('train.txt')

# Build vocabulary
word_to_id, id_to_word = mapping(tokens)

# Generate skip-gram training pairs
X, y = generate_training_data(tokens, word_to_id, window=5)

# Model parameters
vocab_size = len(word_to_id)
embedding_dim = 30
model = init_network(vocab_size, embedding_dim)

# Training
n_iter = 2500
learning_rate = 0.0005
losses = []

print("Starting training...")
for i in range(n_iter):
    loss = backward(model, X, y, learning_rate)
    losses.append(loss)
    if i % 100 == 0:
        print(f"iter: {i:4d} - loss: {loss:.2f}")

# Plot loss
plt.figure(figsize=(8, 5))
plt.plot(losses, color="skyblue")
plt.title("Training Loss (Cross-Entropy)")
plt.xlabel("Iteration")
plt.ylabel("Total Loss")
plt.grid(True)
plt.show()

# --------------------- PCA Visualization ---------------------
embedding_matrix = model["w1"]
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embedding_matrix)

plt.figure(figsize=(16, 12))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7, s=60)

# Annotate only frequent/common words to avoid clutter
word_freq = np.bincount([word_to_id[t] for t in tokens if t in word_to_id])
frequent_words = np.where(word_freq > 2)[0]  # words appearing >10 times

for i in frequent_words:
    word = id_to_word[i]
    plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                 fontsize=9, alpha=0.9, ha='center')

plt.title("Word Embeddings (PCA 2D) - Trained on E-commerce Reviews")
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()

# --------------------- SAVE THE TRAINED EMBEDDINGS ---------------------
print("\nSaving trained embeddings and vocabulary...")

# Method 1: NumPy binary (Recommended - fast & compact)
np.save('ecommerce_word2vec_embeddings.npy', model["w1"])
np.save('ecommerce_word_to_id.npy', word_to_id)
np.save('ecommerce_id_to_word.npy', id_to_word)

print("Saved:")
print("  - ecommerce_word2vec_embeddings.npy   (embedding matrix)")
print("  - ecommerce_word_to_id.npy            (word → id dict)")
print("  - ecommerce_id_to_word.npy            (id → word dict)")

# Optional: Save in text format (like original word2vec)
with open('ecommerce_word2vec.txt', 'w', encoding='utf-8') as f:
    f.write(f"{vocab_size} {embedding_dim}\n")
    for word, idx in word_to_id.items():
        vec = ' '.join(f"{model['w1'][idx][j]:.6f}" for j in range(embedding_dim))
        f.write(f"{word} {vec}\n")
print("  - ecommerce_word2vec.txt              (human-readable format)")

print("\nTraining complete! Embeddings saved. Ready for use in LSTM sentiment analysis.")