import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import re
from transformers import AutoTokenizer, AutoModel
import torch

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
nlp = spacy.load("en_core_web_sm")

file_path = 'Dataset.csv'
df = pd.read_csv(file_path)

def get_counts(column):
  filtered_df = df[df['generated'] == column]
  vectorizer = CountVectorizer(stop_words='english')
  X = vectorizer.fit_transform(filtered_df['text'])
  word_counts = X.sum(axis=0).A1
  words = vectorizer.get_feature_names_out()
  word_freq = dict(zip(words, word_counts))
  sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
  return sorted_word_freq

def is_ai_word(word):
    ai_related_tags = ['VB', 'RB', 'JJ'] 
    doc = nlp(word)
    return doc[0].tag_ in ai_related_tags

def calculate_ai_ratio(text, result):
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text.lower()) 
    ai_count = sum(1 for token in tokens if token in result)  
    return ai_count / len(tokens)

sorted_word_freq = get_counts('1')
sorted_word_freq1 = get_counts('0')
result = []
for word, freq in sorted_word_freq:
  found = False
  for word1, freq1 in sorted_word_freq1:
    if word == word1:
      result.append((word, freq / freq1))
      found = True
      break
  if not found:
    result.append((word, 100))
result = [(word, freq / freq1) if found else (word, 100) for word, freq in sorted_word_freq for word1, freq1 in sorted_word_freq1 if word == word1 or not (found := True)]
filtered_result = [item for item in result if item[1] >= 1.5]
sorted_result = sorted(filtered_result, key=lambda x: x[1], reverse=True)
result = [i for i, score in sorted_result]
result = [i for i in result if not is_ai_word(i)]
df['ai_ratio'] = df['text'].apply(lambda x: calculate_ai_ratio(x, result))