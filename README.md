# tagalog-fake-news-detection

**Compact Transformer Models + Lightweight Ensemble for Low-Resource NLP**

---

## Project Overview

This project benchmarks **compact transformer models** for detecting fake news in **Tagalog**, a low-resource language.

- Addresses the challenge of **fake news detection** in languages with limited labeled datasets.
- Focuses on **efficient models** suitable for deployment in resource-constrained environments.
- Implements **ensemble learning** to improve predictive performance using a weighted soft voting strategy.

---

## Dataset

- **Fake News Filipino Dataset (Cruz et al., 2020)**
- Contains **3,206 articles**: 1,603 real / 1,603 fake
- Sources:
  - Fake: VeraFiles + NUJP
  - Real: Mainstream news outlets
- Preserves **punctuation, capitalization, and code-switching**
- Stored in `data/` directory

---

## Models

This project evaluates the following **compact transformer models**:

| Model          | Type         | Notes                       |
|----------------|-------------|-----------------------------|
| DistilBERT     | Transformer | Lightweight BERT variant    |
| TinyBERT       | Transformer | Pruned BERT model           |
| MobileBERT     | Transformer | Mobile-optimized BERT       |
| MiniLMv2       | Transformer | Efficient BERT alternative |
| ELECTRA-small  | Transformer | Discriminator-style BERT    |

**Ensemble Method:**
- Top 2 models based on F1 score
- Weighted soft voting combining model probabilities

---

## Project Structure

The project directories:

- `src/` — Python scripts & notebooks  
- `data/` — Dataset CSVs or samples  
- `models/` — Trained model checkpoints / ONNX files  
- `figures/` — Plots (ROC, confusion matrix, F1 comparison)  
- `docs/` — Manuscript, daily logs, tables  
- `README.md` — This file  
