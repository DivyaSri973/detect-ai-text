# LLM - Detect AI Generated Text

## Project Overview

LLM - Detect AI Generated Text is a machine learning project that distinguishes between human-written and AI-generated essays. We utilize advanced linguistic analysis, feature engineering, and multiple classification models to identify subtle differences in writing styles.

Our dataset comprises over 100,000 essays sourced from Kaggle and Hugging Face, including texts from middle and high school students as well as large language models (LLMs). Through comprehensive preprocessing, principal component analysis (PCA), and feature selection, we build robust models that accurately detect AI-generated content.

---

## Key Achievements

- Developed detection models using **SVM**, **XGBoost**, and **MLP**, leveraging features such as perplexity, readability indices, and compression factors.
- Achieved **93% accuracy** and strong **AUC-ROC scores** in distinguishing AI-generated text.
- Aggregated and preprocessed large-scale datasets, implemented extensive feature engineering, and performed PCA-based dimensionality reduction and hyperparameter tuning for optimal results.

---

## Features & Analysis

### 1. **Dataset Composition**
- Essays are labeled as either "human-written" or "AI-generated."
- Data aggregated from multiple sources, shuffled, and preprocessed for consistency.

### 2. **Textual Features**
Extracted and engineered features from each essay include:
- **Word Count:** Total number of words.
- **Average Word Length:** Mean character length of words.
- **Vocabulary Richness:** Ratio of unique words to total words.
- **Sentence Length:** Average words per sentence.
- **Punctuation Density:** Number of punctuation marks per word.
- **Misspelled Words:** Total and proportion of misspelled words.
- **Perplexity:** Quantifies how predictable the text is, often lower for AI-generated content.
- **Compression Factors:** Measures the compressibility of text, indicating repetitiveness.

### 3. **Readability & Sentiment**
- **Readability Scores:** Includes Flesch-Kincaid Grade Level, Gunning Fog Index, and SMOG Index.
- **Sentiment Analysis:** Uses TextBlob to derive polarity (positivity/negativity) and subjectivity metrics.

### 4. **Part-of-Speech (POS) Analysis**
- Counts and densities of nouns, verbs, adjectives, and adverbs.
- Normalized POS densities highlight linguistic differences between human and AI-generated writing.

### 5. **Dimensionality Reduction & Feature Selection**
- Applied **PCA** to reduce feature dimensionality and improve model efficiency.
- Selected optimal features for classification using statistical methods.

---

## Setup & Requirements

### Python Libraries

To run the project, ensure the following libraries are installed:

- `pandas` – Data manipulation
- `matplotlib`, `seaborn` – Data visualization
- `nltk` – Tokenization and POS tagging
- `textstat` – Readability metrics
- `textblob` – Sentiment analysis
- `pyspellchecker` – Spell checking
- `scikit-learn` – Data scaling, modeling, and PCA

Install dependencies with:

```bash
pip install pandas matplotlib seaborn nltk textstat textblob pyspellchecker scikit-learn
```

---

## Getting Started

1. Clone the repository and review the dataset and code.
2. Process and analyze essays using the provided scripts.
3. Visualize findings and train models to classify texts as human or AI-generated.
4. Experiment with different feature sets and model hyperparameters for improved detection.

---

## Contributing

We welcome contributions! Please submit issues, pull requests, or suggestions to help improve the project.
