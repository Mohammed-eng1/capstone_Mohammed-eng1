# 🗞️ Fake News Detection — Capstone Project

> **ML, Deep Learning & NLP Applications · Tuwaiq Academy**

---

## 📁 Project Structure

```
capstone_Mohammed-eng1/
├── notebook.ipynb        # Main Jupyter notebook (run top-to-bottom)
├── report.md             # Written report covering all 6 sections
├── README.md             # This file
├── data/
│   ├── True.csv          # Real news articles
│   └── Fake.csv          # Fake news articles
└── plots/                # Auto-generated chart images
    ├── 01_class_balance.png
    ├── 02_confusion_matrices.png
    ├── 03_metrics_comparison.png
    ├── 04_training_curves.png
    ├── 05_full_heatmap.png
    └── 06_radar_f1.png
```

---

## 🎯 Project Goal

Build a complete machine learning pipeline to detect whether a news article is **real or fake** using NLP techniques, classical ML models, and a neural network.

---

## 📊 Dataset

| Detail | Value |
|---|---|
| Source | Kaggle — Fake News Dataset |
| Total Articles | 22,972 |
| Real News | ~10,000 |
| Fake News | ~12,972 |
| Features Used | `title` + `text` (combined) |
| Label | 1 = Real, 0 = Fake |

---

## 🔧 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Create the plots folder
```bash
mkdir plots
```

### 3. Launch the notebook
```bash
jupyter notebook notebook.ipynb
```

> Run all cells **top-to-bottom** — each step depends on the one before it.

---

## 🤖 Models & Results

| Rank | Model | Accuracy | F1-Score |
|---|---|---|---|
| 🥇 | Random Forest | 0.9990 | 0.9994 |
| 🥈 | Gradient Boosting | 0.9984 | 0.9991 |
| 🥉 | Neural Network | 0.9982 | 0.9989 |
| 4 | Logistic Regression | 0.9949 | 0.9970 |
| 5 | KNN | 0.9366 | 0.9636 |

**Best model: Random Forest** with F1-Score = 0.9994 and perfect Recall = 1.000.

---

## 🧠 Pipeline Overview

```
Raw CSV Files
     │
     ▼
 Data Loading & Merging
     │
     ▼
 Text Cleaning (lowercase → remove punctuation → drop stopwords)
     │
     ▼
 TF-IDF Vectorization (5,000 features)
     │
     ▼
 Train / Test Split (80% / 20%, stratified)
     │
     ├──▶ Logistic Regression
     ├──▶ Random Forest          ◀── Best ML Model
     ├──▶ KNN
     ├──▶ Gradient Boosting
     └──▶ Neural Network (Dense + Dropout × 3)
              │
              ▼
         Evaluation & Comparison
```

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
tensorflow
matplotlib
seaborn
nltk
scipy
```

---

## ⚠️ Important Notes

- Always use `fit_transform()` on **training data only** and `transform()` on test data to avoid **data leakage**.
- The `plots/` folder must exist before running the notebook.
- `random_state=42` is used throughout for reproducibility.

---

*Tuwaiq Academy — Bootcamp Programme | ML, Deep Learning & NLP Applications*
