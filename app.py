"""
Tagalog Fake News Detection App
Ensemble (DistilBERT + MobileBERT) with soft voting, attention visualization, and gradient saliency fallback.
"""

import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
from typing import Tuple, Optional, Dict, List


# ============================================================================
# Configuration & Constants
# ============================================================================

MODEL_CONFIG = {
    "distilbert": {
        "path": "jcunado/distilbert-multilingual-fake-news-filipino",
        "f1": 0.9709,
    },
    "mobilebert": {
        "path": "jcunado/MobileBERT-tagalog-fake-news",
        "f1": 0.9647,
    },
}

LABELS = ["Real", "Fake"]  # Official: 0=Real, 1=Fake (jcblaisecruz02 & jcunado)
MAX_LENGTH = 512
TOP_K_TOKENS = 5

# Detect device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# Device & Model Loading
# ============================================================================

@st.cache_resource
def load_models_and_tokenizers() -> Tuple:
    """
    Load both models and tokenizers with proper device placement.
    Returns: (distilbert_model, distilbert_tokenizer, mobilebert_model, mobilebert_tokenizer)
    """
    try:
        distilbert_model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_CONFIG["distilbert"]["path"]
        )
        distilbert_tokenizer = AutoTokenizer.from_pretrained(
            MODEL_CONFIG["distilbert"]["path"]
        )

        mobilebert_model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_CONFIG["mobilebert"]["path"]
        )
        mobilebert_tokenizer = AutoTokenizer.from_pretrained(
            MODEL_CONFIG["mobilebert"]["path"]
        )

        # Move models to device
        distilbert_model.to(DEVICE)
        mobilebert_model.to(DEVICE)

        # Set to evaluation mode
        distilbert_model.eval()
        mobilebert_model.eval()

        return distilbert_model, distilbert_tokenizer, mobilebert_model, mobilebert_tokenizer

    except Exception as e:
        st.error(f"Failed to load models: {str(e)}")
        st.stop()


# ============================================================================
# Ensemble Prediction
# ============================================================================

# ============================================================================
# Ensemble Prediction
# ============================================================================

def compute_ensemble_weights() -> Tuple[float, float]:
    """Normalize F1 scores to ensemble weights."""
    f1_distil = MODEL_CONFIG["distilbert"]["f1"]
    f1_mobile = MODEL_CONFIG["mobilebert"]["f1"]
    total = f1_distil + f1_mobile
    return f1_distil / total, f1_mobile / total


def predict(
    news_article: str,
    distilbert_model,
    distilbert_tokenizer,
    mobilebert_model,
    mobilebert_tokenizer,
) -> Tuple[str, float, Dict]:
    """
    Ensemble prediction with weighted soft voting.
    
    Returns:
        - prediction (str): "Fake" or "Real"
        - confidence (float): Confidence percentage [0, 100]
        - metadata (dict): Contains logits, probabilities, and model-specific info
    """
    w1, w2 = compute_ensemble_weights()

    # Tokenize for both models
    distilbert_inputs = distilbert_tokenizer(
        news_article,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )
    mobilebert_inputs = mobilebert_tokenizer(
        news_article,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )

    # Move inputs to device
    distilbert_inputs = {k: v.to(DEVICE) for k, v in distilbert_inputs.items()}
    mobilebert_inputs = {k: v.to(DEVICE) for k, v in mobilebert_inputs.items()}

    # Forward passes
    with torch.no_grad():
        distilbert_logits = distilbert_model(**distilbert_inputs).logits
        mobilebert_logits = mobilebert_model(**mobilebert_inputs).logits

    # Compute probabilities
    distilbert_probs = torch.softmax(distilbert_logits, dim=1).cpu().numpy()
    mobilebert_probs = torch.softmax(mobilebert_logits, dim=1).cpu().numpy()

    # Ensemble: weighted average of probabilities
    final_probs = w1 * distilbert_probs + w2 * mobilebert_probs
    final_class = np.argmax(final_probs, axis=1)[0]
    final_confidence = np.max(final_probs, axis=1)[0] * 100

    # DEBUG: Print which index is which
    # print(f"DEBUG: final_probs[0] = {final_probs[0]}")
    # print(f"DEBUG: final_class (argmax) = {final_class}")
    # print(f"DEBUG: LABELS[{final_class}] = {LABELS[final_class]}")

    prediction = LABELS[final_class]

    metadata = {
        "distilbert_probs": distilbert_probs[0],
        "mobilebert_probs": mobilebert_probs[0],
        "ensemble_probs": final_probs[0],
        "distilbert_inputs": distilbert_inputs,
        "mobilebert_inputs": mobilebert_inputs,
        "distilbert_tokenizer": distilbert_tokenizer,
        "weights": {"distilbert": w1, "mobilebert": w2},
    }

    return prediction, float(final_confidence), metadata


# ============================================================================
# Attention Visualization
# ============================================================================

def get_attention_scores(
    model,
    inputs: Dict,
    tokenizer,
    model_name: str = "DistilBERT",
) -> Optional[np.ndarray]:
    """
    Extract CLS-to-token attention scores from the last layer.
    
    Returns:
        Array of shape (seq_len,) or None if unavailable.
    """
    try:
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_attentions=True,
                return_dict=True,
            )

        attentions = getattr(outputs, "attentions", None)

        if attentions is None or len(attentions) == 0:
            return None

        # Last layer attention: [batch, heads, seq_len, seq_len]
        last_attention = attentions[-1]
        # CLS-to-token attention averaged across heads
        token_scores = last_attention[0, :, 0, :].mean(dim=0).detach().cpu().numpy()

        return token_scores

    except Exception as e:
        st.warning(f"Attention extraction failed for {model_name}: {str(e)}")
        return None


def get_gradient_saliency(
    model,
    inputs: Dict,
    target_class: int,
) -> Optional[np.ndarray]:
    """
    Compute gradient-based saliency as fallback when attention is unavailable.
    Uses gradient magnitude of embeddings w.r.t. the prediction logit.
    
    Returns:
        Array of shape (seq_len,) or None if computation fails.
    """
    try:
        input_ids = inputs["input_ids"].to(DEVICE)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(DEVICE)

        # Get embeddings with gradient tracking
        # input_ids are integers, so we compute gradients through the embedding layer
        embeddings = model.get_input_embeddings()(input_ids)
        embeddings.requires_grad_(True)

        # Forward pass through model using embeddings
        outputs = model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            return_dict=True,
        )
        logits = outputs.logits

        # Compute loss w.r.t. target class logit
        target_logit = logits[0, target_class]
        target_logit.backward()

        # Gradient magnitude per token (sum across embedding dimension)
        if embeddings.grad is not None:
            grad_magnitude = torch.abs(embeddings.grad).sum(dim=-1)[0].detach().cpu().numpy()
            return grad_magnitude
        else:
            return None

    except Exception as e:
        return None


def highlight_tokens(
    tokens: List[str],
    scores: Optional[np.ndarray],
    method: str = "attention",
) -> str:
    """
    Generate markdown with top-K highlighted tokens.
    
    Args:
        tokens: List of token strings
        scores: Optional array of token importance scores
        method: "attention" or "gradient"
    
    Returns:
        Markdown-formatted string with bold highlights on top-K tokens.
    """
    if scores is None or len(scores) == 0:
        return " ".join(tokens)

    # Ensure scores length matches tokens (handle [CLS], [SEP], padding)
    scores = scores[: len(tokens)]

    # Get top-K indices
    k = min(TOP_K_TOKENS, len(tokens))
    top_indices = set(np.argsort(scores)[-k:])

    # Build highlighted output
    highlighted = [f"**{tokens[i]}**" if i in top_indices else tokens[i] for i in range(len(tokens))]
    return " ".join(highlighted)


# ============================================================================
# UI Components
# ============================================================================

def render_default_state():
    """Render default page when no input is submitted."""
    st.title("Prediction: Waiting for Input")
    st.subheader("Confidence: -%")
    st.write("#### Token Highlighting")
    st.write("*Submit a Tagalog news article to see attention-weighted token highlights.*")
    st.divider()
    st.write(
        "**Model Info:** "
        "[DistilBERT + MobileBERT Ensemble (Soft Voting, F1 Scores)]"
        "(https://github.com/joms-hub/tagalog-fake-news-detection)"
    )


def render_results(
    prediction: str,
    confidence: float,
    metadata: Dict,
    distilbert_model,
    distilbert_tokenizer,
):
    """Render prediction results, model probabilities, and token highlighting."""
    
    # Color-code prediction
    color = "🔴 FAKE" if prediction == "Fake" else "🟢 REAL"
    st.title(f"Prediction: {color}")
    st.subheader(f"Confidence: {confidence:.2f}%")

    # Ensemble probability breakdown
    with st.expander("📊 Ensemble Breakdown"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            distil_real, distil_fake = metadata["distilbert_probs"]  # [class_0=Real, class_1=Fake]
            st.metric("DistilBERT (Real)", f"{distil_real*100:.2f}%")
            st.metric("DistilBERT (Fake)", f"{distil_fake*100:.2f}%")
        
        with col2:
            mobile_real, mobile_fake = metadata["mobilebert_probs"]  # [class_0=Real, class_1=Fake]
            st.metric("MobileBERT (Real)", f"{mobile_real*100:.2f}%")
            st.metric("MobileBERT (Fake)", f"{mobile_fake*100:.2f}%")
        
        with col3:
            ens_real, ens_fake = metadata["ensemble_probs"]  # [class_0=Real, class_1=Fake]
            st.metric("Ensemble (Real)", f"{ens_real*100:.2f}%")
            st.metric("Ensemble (Fake)", f"{ens_fake*100:.2f}%")
        
        st.write(f"**Weights:** DistilBERT {metadata['weights']['distilbert']:.3f} | MobileBERT {metadata['weights']['mobilebert']:.3f}")

    # Token Highlighting with fallback
    st.write("#### 🔍 Token Highlighting")
    
    distilbert_inputs = metadata["distilbert_inputs"]
    tokens = distilbert_tokenizer.convert_ids_to_tokens(distilbert_inputs["input_ids"][0])

    # Try attention first
    attention_scores = get_attention_scores(
        distilbert_model,
        distilbert_inputs,
        distilbert_tokenizer,
        "DistilBERT",
    )

    if attention_scores is not None:
        highlighted_text = highlight_tokens(tokens, attention_scores, method="attention")
        st.write(f"*Top {TOP_K_TOKENS} tokens by attention weight:*")
        st.markdown(highlighted_text)
    else:
        # Fallback: try gradient saliency (silent attempt)
        final_class = 1 if metadata["ensemble_probs"][1] > metadata["ensemble_probs"][0] else 0
        gradient_scores = get_gradient_saliency(
            distilbert_model,
            distilbert_inputs,
            target_class=final_class,
        )

        if gradient_scores is not None:
            highlighted_text = highlight_tokens(tokens, gradient_scores, method="gradient")
            st.write(f"*Top {TOP_K_TOKENS} tokens by gradient magnitude (saliency):*")
            st.markdown(highlighted_text)
        else:
            # Final fallback: just show tokens
            st.write("*Token importance visualization unavailable. Showing raw tokens:*")
            st.write(" ".join(tokens))

    # Debug section to diagnose label inversion
    # with st.expander("🔧 Debug: Raw Model Outputs"):
    #     col1, col2 = st.columns(2)
    #     with col1:
    #         st.write("**DistilBERT Probabilities:**")
    #         d_real, d_fake = metadata["distilbert_probs"]  # [class_0=Real, class_1=Fake]
    #         st.write(f"  Real: {d_real:.4f}")
    #         st.write(f"  Fake: {d_fake:.4f}")
    #     with col2:
    #         st.write("**MobileBERT Probabilities:**")
    #         m_real, m_fake = metadata["mobilebert_probs"]  # [class_0=Real, class_1=Fake]
    #         st.write(f"  Real: {m_real:.4f}")
    #         st.write(f"  Fake: {m_fake:.4f}")
        
    #     st.write("---")
    #     st.write("**Probabilities explained:**")
    #     st.write("Displayed as [Real probability, Fake probability]")
    #     st.write("The ensemble prediction picks the class with highest probability.")

    st.divider()
    st.write(
        "**Model Info:** "
        "[DistilBERT + MobileBERT Ensemble (Soft Voting, F1 Scores)]"
        "(https://github.com/joms-hub/tagalog-fake-news-detection)"
    )


# ============================================================================
# Main App
# ============================================================================

def main():
    st.set_page_config(
        page_title="Tagalog Fake News Detector",
        page_icon="🔍",
        layout="wide",
    )

    # Load models once
    distilbert_model, distilbert_tokenizer, mobilebert_model, mobilebert_tokenizer = load_models_and_tokenizers()

    # Sidebar
    st.sidebar.title("📰 Fake News Detection")
    st.sidebar.write(f"*Device: **{DEVICE.upper()}***")
    
    news_article = st.sidebar.text_area(
        "News Article",
        placeholder="Input a Tagalog news article here...",
        height=150,
    )
    submit_button = st.sidebar.button("Analyze", use_container_width=True)

    # Main content
    if not (submit_button and news_article):
        render_default_state()
    else:
        try:
            # Run prediction
            with st.spinner("🔄 Analyzing article..."):
                prediction, confidence, metadata = predict(
                    news_article,
                    distilbert_model,
                    distilbert_tokenizer,
                    mobilebert_model,
                    mobilebert_tokenizer,
                )

            # Render results
            render_results(
                prediction,
                confidence,
                metadata,
                distilbert_model,
                distilbert_tokenizer,
            )

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.write("Please try again or contact the developer.")


if __name__ == "__main__":
    main()