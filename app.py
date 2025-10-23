import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np

# Sidebar layout
st.sidebar.title("Fake News Detection")
news_article = st.sidebar.text_area("News Article", placeholder="Input a Tagalog news article here...")
submit_button = st.sidebar.button("Submit")

# Paths to models
distilbert_model_path = "jcunado/distilbert-multilingual-fake-news-filipino"
mobilebert_model_path = "jcunado/MobileBERT-tagalog-fake-news"

# F1 scores
F1_DistilBERT = 0.9709
F1_MobileBERT = 0.9647

# Weights for ensemble
w1 = F1_DistilBERT / (F1_DistilBERT + F1_MobileBERT)
w2 = F1_MobileBERT / (F1_DistilBERT + F1_MobileBERT)

# Load models and tokenizers
@st.cache_resource
def load_models_and_tokenizers():
    distilbert_model = AutoModelForSequenceClassification.from_pretrained(distilbert_model_path)
    distilbert_tokenizer = AutoTokenizer.from_pretrained(distilbert_model_path)

    mobilebert_model = AutoModelForSequenceClassification.from_pretrained(mobilebert_model_path)
    mobilebert_tokenizer = AutoTokenizer.from_pretrained(mobilebert_model_path)

    return distilbert_model, distilbert_tokenizer, mobilebert_model, mobilebert_tokenizer

distilbert_model, distilbert_tokenizer, mobilebert_model, mobilebert_tokenizer = load_models_and_tokenizers()

# Display default main panel before submit
if not (submit_button and news_article):
    st.title("Prediction: Waiting for Input")
    st.subheader("Confidence: -%")
    st.write("Token Highlighting")
    st.write("Model Info: [DistilBERT + MobileBERT Ensemble (Soft Voting, F1 Scores)](https://github.com/joms-hub/tagalog-fake-news-detection)")

# Process input and display results only once upon submit
if submit_button and news_article:
    # Tokenize inputs
    distilbert_inputs = distilbert_tokenizer(news_article, return_tensors="pt", truncation=True, padding=True)
    mobilebert_inputs = mobilebert_tokenizer(news_article, return_tensors="pt", truncation=True, padding=True)

    # Predict with models
    with torch.no_grad():
        distilbert_logits = distilbert_model(**distilbert_inputs).logits
        mobilebert_logits = mobilebert_model(**mobilebert_inputs).logits

    # Softmax probabilities
    distilbert_probs = torch.softmax(distilbert_logits, dim=1).numpy()
    mobilebert_probs = torch.softmax(mobilebert_logits, dim=1).numpy()

    # Ensemble probabilities
    final_probs = w1 * distilbert_probs + w2 * mobilebert_probs
    final_class = np.argmax(final_probs, axis=1)
    final_confidence = np.max(final_probs, axis=1) * 100

    # Labels
    labels = ["Fake", "Real"]
    prediction = labels[final_class[0]]
    confidence = round(final_confidence[0])

    # Display results
    st.title(f"Prediction: {prediction}")
    st.subheader(f"Confidence: {confidence}%")

    # Token Highlighting
    # Extract attention scores (example with DistilBERT as placeholder)
    with torch.no_grad():
        distilbert_attention = distilbert_model(**distilbert_inputs, output_attentions=True).attentions[-1]
    distilbert_attention = distilbert_attention.mean(dim=1).detach().numpy()  # Aggregate across heads
    token_scores = distilbert_attention.mean(axis=0)  # Aggregate across tokens
    tokens = distilbert_tokenizer.convert_ids_to_tokens(distilbert_inputs["input_ids"][0])

    # Highlight top N tokens
    N = 5
    top_indices = np.argsort(token_scores)[-N:]
    highlighted_tokens = [f"**{tokens[i]}**" if i in top_indices else tokens[i] for i in range(len(tokens))]
    st.write("Token Highlighting:")
    st.write(" ".join(highlighted_tokens))

    st.write("Model Info: [DistilBERT + MobileBERT Ensemble (Soft Voting, F1 Scores)](https://github.com/joms-hub/tagalog-fake-news-detection)")