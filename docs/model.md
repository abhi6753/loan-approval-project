# Deep Learning Model Documentation

This document describes the design, architectures, loss functions, evaluation metrics, and academic scope for **Unit IV (LSTM Classifier)** and **Unit V (Autoencoder Anomaly Detection)** in the Loan Approval Prediction project.

---

## 1. Unit IV — Sequential LSTM Binary Classifier

### 1.1 Academic Scope & Dataset Limitation
- **Dataset Nature**: The standard Kaggle Loan Prediction Dataset (`data/loan_data.csv`) is tabular with one row per applicant (381 records, 12 features).
- **Sequence Adaptation**: Because the dataset is non-temporal, tabular feature vectors $X \in \mathbb{R}^{D}$ (where $D$ is the number of preprocessed feature columns) are reshaped into 3D tensors:
  $$\text{Shape}: (\text{batch\_size}, \text{timesteps}=D, \text{features}=1)$$
- **Academic Purpose**: This allows sequence processing over multi-dimensional feature columns using recurrent LSTM memory cells to fulfill the Unit IV syllabus requirement without making false claims about temporal financial history.

### 1.2 Model Architecture
```
Input Tensor: (D, 1)
     ↓
LSTM Layer (32 units, return_sequences=False)
     ↓
Dropout Layer (rate = 0.20)
     ↓
Dense Layer (16 units, activation = 'relu')
     ↓
Dense Output Layer (1 unit, activation = 'sigmoid')
```

### 1.3 Compilation & Loss
- **Optimizer**: Adam ($\text{learning\_rate} = 0.001$)
- **Loss Function**: Binary Cross-Entropy
  $$\mathcal{L}_{\text{BCE}} = - \frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$
- **Decision Threshold**: $\hat{y}_i \ge 0.50 \implies \text{Approved (1)}$, else $\text{Rejected (0)}$.

---

## 2. Unit V — Autoencoder Anomaly Detection

### 2.1 Concept & Architecture
The Autoencoder is trained unsupervised on training data ($X_{\text{train}} \to X_{\text{train}}$) to learn a compressed representation of typical loan applicant feature distributions.

```
Input Vector X ∈ ℝ^D
     ↓
Encoder: Dense(32, relu) → Dense(16, relu) → Latent Bottleneck Z ∈ ℝ^8
     ↓
Decoder: Dense(16, relu) → Dense(32, relu) → Reconstructed Vector X̂ ∈ ℝ^D
```

### 2.2 Reconstruction Error & Anomaly Threshold
- **Loss Function**: Mean Squared Error (MSE)
  $$\text{MSE}(X, \hat{X}) = \frac{1}{D} \sum_{j=1}^D (X_j - \hat{X}_j)^2$$
- **Threshold Calculation**: Calculated strictly on $X_{\text{train}}$ reconstruction errors:
  $$\tau_{\text{anomaly}} = \mu_{\text{train\_error}} + 2.0 \cdot \sigma_{\text{train\_error}}$$
- **Anomaly Flag**:
  $$\text{Anomaly\_Flag} = \begin{cases} \text{True ("Requires Review")}, & \text{if } \text{MSE} > \tau_{\text{anomaly}} \\ \text{False ("Normal Distribution")}, & \text{otherwise} \end{cases}$$
- **Interpretation**: An anomaly flag indicates that an applicant's financial metrics deviate significantly from the baseline training distribution. It serves as an academic reconstruction flag, NOT proof of fraud or automatic loan rejection.

---

## 3. Empirical Evaluation Metrics
All metrics are generated from actual execution on the test set split:
- **Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$
- **Precision**: $\frac{TP}{TP + FP}$
- **Recall**: $\frac{TP}{TP + FN}$
- **F1 Score**: $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
- **Confusion Matrix**: True Positive, True Negative, False Positive, False Negative breakdown.
