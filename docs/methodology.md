# Chapter 4: Design and Methodology

## 4.1 Sources of Data
- **Dataset**: Fake News Filipino dataset (Cruz et al., 2020)  
- **Size**: 3,206 articles (1,603 real / 1,603 fake)  
- **Sources**:  
  - Fake: VeraFiles, NUJP  
  - Real: Mainstream outlets  
- **Characteristics**: Preserves punctuation, capitalization, and code-switching  

---

## 4.2 Research Procedure

### 4.2.1 Data Gathering
- Dataset verified via fact-checking organizations  

### 4.2.2 Treatment of Data
- Split: 70% training / 15% validation / 15% testing  
- Minimal preprocessing  
- Tokenizer per model  
- Truncation and padding to 512 tokens  

---

## 4.3 Conceptual Framework
- Fine-tuning compact models with transfer learning  
- **Hyperparameters**:  
  - Batch size: 32  
  - Learning rate: 2e-5  
  - Optimizer: Adam  
- Early stopping (patience = 2)  
- Ensemble: Top 2 models by F1 score → weighted soft voting  

---

## 4.4 Analysis and Design
- Ensemble equations for weighted averaging  
- Probability combination of softmax outputs: *(w₁, w₂)*  

---

## 4.5 Development Model
- Rapid Prototyping model for iterative refinement  

---

## 4.6 Development Approaches
- Fine-tuning compact transformers with transfer learning  
- Max 10 epochs  
- Batch size: 32  
- Optimizer: Adam  
- Early stopping  
- Ensemble integration at the end  

---

## 4.7 Software Development Tools
- Python  
- HuggingFace  
- scikit-learn  
- NumPy  
- pandas  
- Colab Pro  
- matplotlib / seaborn  
- GitHub  

---

## 4.8 Project Management

### 4.8.1 Schedule
- Gantt chart from June–October (topic selection → conference)  

### 4.8.2 Responsibilities
- 2 researchers + 1 adviser  

### 4.8.3 Budget
- Total: ₱195,625  
  - Equipment  
  - Utilities  
  - Conference  
  - Labor  

---

## 4.9 Verification, Validation, and Testing
- **Metrics**: Accuracy, precision, recall, F1, binary cross-entropy loss  
- **Tools**: Confusion matrix, ROC curve, AUC  
- **Efficiency Metrics**: Inference latency, peak RAM usage, model size  
- **Deployment**: Models exported to ONNX for cross-platform evaluation  
