# 🗞️ Fake News Detection — Capstone Report
**ML, Deep Learning & NLP Applications · Tuwaiq Academy**

---

## 1. Introduction

I chose the **Fake News Detection** dataset because detecting misinformation automatically is one of the most impactful real-world applications of Natural Language Processing. The goal is to build models that can classify news articles as either **real (1)** or **fake (0)** based on the article's title and body text.

---

## 2. Dataset Description

| Detail | Value |
|---|---|
| Source | Kaggle — Fake News Dataset |
| Columns | `title`, `text`, `subject`, `date`, `label` |
| Total Rows | 22,972 articles |
| Real News | ~10,000 |
| Fake News | ~12,972 |
| Missing Values | None |

The two classes are nearly balanced. Both **Accuracy** and **F1-Score** were used as evaluation metrics.

---

## 3. Models Used

### Data Preparation Steps
- Combined `title` and `text` into a single column
- Cleaned text: lowercasing, removing punctuation and numbers, dropping stop words
- Converted text to numbers using **TF-IDF** (max 5,000 features)
- Split data: 80% train / 20% test (`random_state=42`, `stratify=y`)

### Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Random Forest | 0.9990 | 0.9988 | 1.0000 | 0.9994 |
| Gradient Boosting | 0.9984 | 0.9991 | 0.9991 | 0.9991 |
| Logistic Regression | 0.9949 | 0.9942 | 0.9998 | 0.9970 |
| KNN | 0.9366 | 0.9406 | 0.9876 | 0.9636 |

> **Best Model: Random Forest — F1-Score = 0.9994**
>
> Random Forest combines hundreds of independent decision trees that vote on the final result, making it robust against noise in text data. It also achieved a perfect Recall of 1.000, meaning it missed zero real news articles.

---

## 4. Neural Network

A 4-layer Dense network was built with the following architecture:

| Layer | Details |
|---|---|
| Dense 1 | 256 neurons, ReLU |
| Dropout | 0.4 |
| Dense 2 | 128 neurons, ReLU |
| Dropout | 0.3 |
| Dense 3 | 64 neurons, ReLU |
| Dropout | 0.2 |
| Output | 1 neuron, Sigmoid |

- **Optimizer:** Adam (lr = 0.001)
- **Loss:** Binary Crossentropy
- **Epochs:** 20 | **Batch Size:** 128

| Metric | Score |
|---|---|
| Accuracy | 0.9982 |
| Precision | 0.9988 |
| Recall | 0.9991 |
| F1-Score | **0.9989** |

Training curves showed smooth convergence with no significant overfitting, indicating that the Dropout layers worked effectively.

---

## 5. Results & Comparison

| Rank | Model | F1-Score |
|---|---|---|
| 🥇 | Random Forest | 0.9994 |
| 🥈 | Gradient Boosting | 0.9991 |
| 🥉 | Neural Network | 0.9989 |
| 4 | Logistic Regression | 0.9970 |
| 5 | KNN | 0.9636 |

The Neural Network ranked third with only a marginal gap behind the top two models, but required significantly more time to train. Random Forest is the best overall choice for this task, balancing accuracy and efficiency. KNN performed worst because it relies on distance calculations in a 5,000-dimensional feature space, which weakens its performance.

---

## 6. What I Learned

- **Text preprocessing was the single most impactful step** — without cleaning, model performance drops significantly.
- More complex models are not always better; Logistic Regression achieved F1 = 0.9970 with far fewer resources than the neural network.
- Random Forest outperformed all other models due to its ability to handle high-dimensional text features effectively.
- Monitoring training and validation curves is essential for detecting overfitting early.
- Using `fit_transform` on training data only and `transform` on test data prevents **Data Leakage** and ensures honest model evaluation.
- KNN suffers from the **Curse of Dimensionality** — the more features, the weaker its performance becomes.

---

*Tuwaiq Academy — Bootcamp Programme | ML, Deep Learning & NLP Applications*
