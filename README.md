## Overview

Our project analyzes a dataset of essays to distinguish between human-written and AI-generated texts. The analysis involves exploring text features, calculating linguistic metrics, and leveraging machine learning techniques to detect patterns indicative of text authorship.

The processed dataset includes essays from middle and high school students, as well as those generated by various language models. A variety of text processing, feature engineering, and visualization techniques have been applied to derive meaningful insights.


## Features and Analysis

### 1. **Dataset Information**
- The dataset contains text essays labeled as "human-written" or "AI-generated."
- Combined data from multiple sources, shuffled and preprocessed to ensure uniformity.

### 2. **Text Features**
Key features engineered from the text include:
- **Word Count**: Total number of words in an essay.
- **Average Word Length**: Average length of words in an essay.
- **Vocabulary Richness**: Ratio of unique words to total words.
- **Sentence Length**: Average number of words per sentence.
- **Punctuation Density**: Punctuation marks per word.
- **Misspelled Word Count**: Total and relative count of misspelled words.

### 3. **Readability and Sentiment**
- **Readability Scores**: Includes Flesch-Kincaid Grade Level, Gunning Fog Index, and SMOG Index.
- **Sentiment Analysis**: Polarity and subjectivity metrics derived using TextBlob.

### 4. **Part-of-Speech (POS) Analysis**
- Counts and densities of nouns, verbs, adjectives, and adverbs.
- Normalized POS densities highlight linguistic differences between human and AI-generated texts.

## Requirements

### Python Libraries
- `pandas`: For data manipulation.
- `matplotlib` and `seaborn`: For visualization.
- `nltk`: For POS tagging and tokenization.
- `textstat`: For readability metrics.
- `textblob`: For sentiment analysis.
- `spellchecker`: For detecting misspelled words.
- `sklearn`: For scaling and further data preparation.

Install the required libraries using:
```bash
pip install pandas matplotlib seaborn nltk textstat textblob pyspellchecker scikit-learn
```
