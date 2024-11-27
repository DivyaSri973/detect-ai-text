#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:55:34 2024

@author: ashnaarora
"""

import pandas as pd


file_path = '/Users/ashnaarora/Downloads/Dataset.csv'
df = pd.read_csv(file_path)


print(df.head())


#%%

df.shape


#%%

import pandas as pd
import matplotlib.pyplot as plt


df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
df['average_word_length'] = df['text'].apply(lambda x: sum(len(word) for word in str(x).split()) / len(str(x).split()) if len(str(x).split()) > 0 else 0)

# Plot Word Count Distribution
plt.figure(figsize=(10, 6))
plt.hist(df['word_count'], bins=50, alpha=0.7, color='blue', edgecolor='black')
plt.title('Distribution of Word Count')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.show()

# Plot Average Word Length Distribution
plt.figure(figsize=(10, 6))
plt.hist(df['average_word_length'], bins=50, alpha=0.7, color='green', edgecolor='black')
plt.title('Distribution of Average Word Length')
plt.xlabel('Average Word Length')
plt.ylabel('Frequency')
plt.show()

#%%
# Inspect essays with unusually high average word lengths (e.g., greater than 20)
unusual_avg_length = df[df['average_word_length'] > 20]
print(f"Essays with high average word lengths:\n{unusual_avg_length[['essay_id', 'text', 'average_word_length']]}")

#%%
# Inspect the problematic essay
problematic_essay = df[df['essay_id'] == 'CDABA09822C7']['text'].values[0]
print(f"Text of the problematic essay:\n{problematic_essay}")

#%%
df = df[df['average_word_length'] <= 20]

#%%

df.shape
#%%

df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
df['average_word_length'] = df['text'].apply(lambda x: sum(len(word) for word in str(x).split()) / len(str(x).split()) if len(str(x).split()) > 0 else 0)

# Plot Word Count Distribution
plt.figure(figsize=(10, 6))
plt.hist(df['word_count'], bins=50, alpha=0.7, color='blue', edgecolor='black')
plt.title('Distribution of Word Count')
plt.xlabel('Word Count')
plt.ylabel('Frequency')
plt.show()

# Plot Average Word Length Distribution
plt.figure(figsize=(10, 6))
plt.hist(df['average_word_length'], bins=50, alpha=0.7, color='green', edgecolor='black')
plt.title('Distribution of Average Word Length')
plt.xlabel('Average Word Length')
plt.ylabel('Frequency')
plt.show()

#%%
    
distinct_values = df['source'].unique()

# Display distinct values
print("Distinct values in the 'source' column:")
for value in distinct_values:
    print(value)
    
#%%
# Rename specific values in the 'source' column
rename_mapping = {
    'chat_gpt_moth': 'gpt-4',
    'original_moth': 'human',
    'persuade_corpus': 'human',
    'falcon_180b_v1': 'falcon_180b',
    'llama_70b_v1': 'llama_70b',
    'radek_500': 'gpt-3.5-turbo',
    'train_essays': 'human'
}

# Apply the renaming
# Explicitly use .loc to avoid SettingWithCopyWarning
df.loc[:, 'source'] = df['source'].replace(rename_mapping)

# Display updated distinct values to confirm changes
distinct_values_updated = df['source'].unique()
print("Updated distinct values in the 'source' column:")
for value in distinct_values_updated:
    print(value)

#%%
import matplotlib.pyplot as plt

# Count the frequency of each unique value in the 'source' column
source_counts = df['source'].value_counts()

# Plot a bar chart
plt.figure(figsize=(10, 6))
source_counts.plot(kind='bar', color='purple', edgecolor='black')
plt.title('Frequency of Sources', fontsize=16)
plt.xlabel('Source', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.tight_layout()
plt.show()

#%%

# Calculate Vocabulary Richness
df['vocabulary_richness'] = df['text'].apply(lambda x: len(set(str(x).split())) / len(str(x).split()) if len(str(x).split()) > 0 else 0)

# Display first few rows to check the new column
print(df[['essay_id', 'vocabulary_richness']].head())

# Plot Vocabulary Richness Distribution
plt.figure(figsize=(10, 6))
plt.hist(df['vocabulary_richness'], bins=50, alpha=0.7, color='orange', edgecolor='black')
plt.title('Distribution of Vocabulary Richness')
plt.xlabel('Vocabulary Richness (Unique Words / Total Words)')
plt.ylabel('Frequency')
plt.show()

# Inspect texts with unusually high richness (e.g., > 0.9)
high_richness = df[df['vocabulary_richness'] > 0.9]
print("Texts with very high vocabulary richness:")
print(high_richness[['essay_id', 'text', 'vocabulary_richness']])



#%%
from textstat import textstat
from textblob import TextBlob
from joblib import Parallel, delayed
import pandas as pd

# Function to calculate readability metrics for a batch
def process_batch(batch):
    results = []
    for text in batch:
        results.append({
            'flesch_kincaid': textstat.flesch_kincaid_grade(text),
            'gunning_fog': textstat.gunning_fog(text),
            'smog_index': textstat.smog_index(text),
            'polarity': TextBlob(text).sentiment.polarity,
            'subjectivity': TextBlob(text).sentiment.subjectivity
        })
    return results

# Process data in batches of 1000
batch_size = 1000
readability_sentiment_results = []

for i in range(0, len(df), batch_size):
    batch = df['text'][i:i + batch_size]
    results = process_batch(batch)
    readability_sentiment_results.extend(results)

# Convert results to a DataFrame
results_df = pd.DataFrame(readability_sentiment_results)
results_df

#%%

# Concatenate results with the original DataFrame
df = pd.concat([df.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)

# Display the updated DataFrame
print(df.head())


#%%

# List of metrics to visualize
import matplotlib.pyplot as plt

metrics = ['flesch_kincaid', 'gunning_fog', 'smog_index']

for metric in metrics:
    plt.figure(figsize=(8, 5))
    df[metric].plot(kind='hist', bins=30, color='lightgreen', edgecolor='black')
    plt.title(f'Distribution of {metric.capitalize()}')
    plt.xlabel(metric.capitalize())
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    
#%%


for metric in ['polarity', 'subjectivity']:
    plt.figure(figsize=(8, 5))
    df[metric].plot(kind='hist', bins=30, color='lightgreen', edgecolor='black', title=f'Distribution of {metric}')
    plt.xlabel(metric.capitalize())
    plt.ylabel('Frequency')
    plt.show()

#%%

# Filter texts with very high readability scores
high_readability = df[(df['flesch_kincaid'] > 50) | (df['gunning_fog'] > 50)]

# Display the results
print("Texts with very high readability scores:")
print(high_readability[['source', 'text', 'flesch_kincaid', 'gunning_fog']])

# Save these texts for further inspection
high_readability.to_csv('high_readability_texts.csv', index=False)
print("Saved high readability texts to 'high_readability_texts.csv'")

#%%
# Count the sources of high readability texts
high_readability_source_counts = high_readability['source'].value_counts()

# Plot the distribution of sources for high readability texts
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
high_readability_source_counts.plot(kind='bar', color='purple', edgecolor='black')
plt.title('Source Distribution for High Readability Texts', fontsize=16)
plt.xlabel('Source', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.tight_layout()
plt.show()

#%%
# Display texts with very high readability scores
for index, row in high_readability.iterrows():
    print(f"Index: {index}")
    print(f"Source: {row['source']}")
    print(f"Flesch-Kincaid: {row['flesch_kincaid']}")
    print(f"Gunning Fog: {row['gunning_fog']}")
    print(f"Text: {row['text']}")
    print("-" * 100)


#%%
from spellchecker import SpellChecker

# Initialize the SpellChecker
spell = SpellChecker()

# Function to count misspelled words in a text
def count_misspelled_words(text):
    words = str(text).split()  # Split text into words
    misspelled = spell.unknown(words)  # Find misspelled words
    return len(misspelled)

# Apply the function to the dataset
df['misspelled_word_count'] = df['text'].apply(count_misspelled_words)

# Display the first few rows with misspelled word counts
print(df[['essay_id', 'misspelled_word_count']].head())

# Plot the distribution of misspelled word counts
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(df['misspelled_word_count'], bins=50, alpha=0.7, color='red', edgecolor='black')
plt.title('Distribution of Misspelled Word Counts')
plt.xlabel('Misspelled Word Count')
plt.ylabel('Frequency')
plt.show()

#%%
# Group by 'source' and sum up the misspelled word counts
misspelled_by_source = df.groupby('source')['misspelled_word_count'].sum()

# Display the result
print("Count of misspelled words by source:")
print(misspelled_by_source)

# Plot the results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
misspelled_by_source.plot(kind='bar', color='orange', edgecolor='black')
plt.title('Misspelled Word Counts by Source', fontsize=16)
plt.xlabel('Source', fontsize=12)
plt.ylabel('Total Misspelled Word Count', fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.tight_layout()
plt.show()

#%%
df.columns

#%%

df['sentence_length'] = df['text'].apply(
    lambda x: sum(len(sentence.split()) for sentence in str(x).split('.')) / len(str(x).split('.')) if len(str(x).split('.')) > 0 else 0
)

# Display the first few rows to check the new column
print(df[['essay_id', 'sentence_length']].head())

# Plot the distribution of sentence lengths
plt.figure(figsize=(10, 6))
plt.hist(df['sentence_length'], bins=50, alpha=0.7, color='blue', edgecolor='black')
plt.title('Distribution of Sentence Length')
plt.xlabel('Average Sentence Length (words per sentence)')
plt.ylabel('Frequency')
plt.show()


#%%

import string

# Calculate punctuation usage as the total count of punctuation marks per text
df['punctuation_count'] = df['text'].apply(
    lambda x: sum(1 for char in str(x) if char in string.punctuation)
)

# Calculate punctuation density (punctuation marks per word)
df['punctuation_density'] = df.apply(
    lambda row: row['punctuation_count'] / row['word_count'] if row['word_count'] > 0 else 0,
    axis=1
)

# Display the first few rows to check the new columns
print(df[['essay_id', 'punctuation_count', 'punctuation_density']].head())

# Plot the distribution of punctuation counts
plt.figure(figsize=(10, 6))
plt.hist(df['punctuation_count'], bins=50, alpha=0.7, color='blue', edgecolor='black')
plt.title('Distribution of Punctuation Count')
plt.xlabel('Punctuation Count')
plt.ylabel('Frequency')
plt.show()

# Plot the distribution of punctuation density
plt.figure(figsize=(10, 6))
plt.hist(df['punctuation_density'], bins=50, alpha=0.7, color='green', edgecolor='black')
plt.title('Distribution of Punctuation Density')
plt.xlabel('Punctuation Density (Punctuation Marks per Word)')
plt.ylabel('Frequency')
plt.show()

# Inspect texts with very high punctuation density (e.g., > 0.5)
high_punctuation_density = df[df['punctuation_density'] > 0.5]
print("Texts with very high punctuation density:")
print(high_punctuation_density[['essay_id', 'text', 'punctuation_density']])


#%%

import nltk
from collections import Counter

# Download required resources for nltk (only the first time)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Function to calculate POS statistics using nltk
def calculate_pos_stats_nltk(text):
    tokens = nltk.word_tokenize(str(text))  # Tokenize text
    pos_tags = nltk.pos_tag(tokens)         # Get POS tags
    pos_counts = Counter(tag for word, tag in pos_tags)  # Count POS tags
    total_tokens = len(tokens)
    # Normalize counts by total tokens for density
    pos_density = {pos: count / total_tokens for pos, count in pos_counts.items()}
    return pos_counts, pos_density

# Apply the function to calculate POS stats
df['pos_counts'], df['pos_density'] = zip(*df['text'].apply(calculate_pos_stats_nltk))

# Extract specific POS counts and densities into separate columns (e.g., Nouns, Verbs, Adjectives)
# Map POS tags to broader categories for analysis
noun_tags = ['NN', 'NNS', 'NNP', 'NNPS']
verb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
adjective_tags = ['JJ', 'JJR', 'JJS']
adverb_tags = ['RB', 'RBR', 'RBS']

df['noun_count'] = df['pos_counts'].apply(lambda x: sum(x[tag] for tag in noun_tags if tag in x))
df['verb_count'] = df['pos_counts'].apply(lambda x: sum(x[tag] for tag in verb_tags if tag in x))
df['adjective_count'] = df['pos_counts'].apply(lambda x: sum(x[tag] for tag in adjective_tags if tag in x))
df['adverb_count'] = df['pos_counts'].apply(lambda x: sum(x[tag] for tag in adverb_tags if tag in x))

df['noun_density'] = df['pos_density'].apply(lambda x: sum(x[tag] for tag in noun_tags if tag in x))
df['verb_density'] = df['pos_density'].apply(lambda x: sum(x[tag] for tag in verb_tags if tag in x))
df['adjective_density'] = df['pos_density'].apply(lambda x: sum(x[tag] for tag in adjective_tags if tag in x))
df['adverb_density'] = df['pos_density'].apply(lambda x: sum(x[tag] for tag in adverb_tags if tag in x))

# Display the first few rows to check the new columns
print(df[['essay_id', 'noun_count', 'verb_count', 'adjective_count', 'adverb_count']].head())
print(df[['essay_id', 'noun_density', 'verb_density', 'adjective_density', 'adverb_density']].head())

# Plot the distribution of noun density
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(df['noun_density'], bins=50, alpha=0.7, color='blue', edgecolor='black')
plt.title('Distribution of Noun Density')
plt.xlabel('Noun Density')
plt.ylabel('Frequency')
plt.show()

# Plot the distribution of verb density
plt.figure(figsize=(10, 6))
plt.hist(df['verb_density'], bins=50, alpha=0.7, color='green', edgecolor='black')
plt.title('Distribution of Verb Density')
plt.xlabel('Verb Density')
plt.ylabel('Frequency')
plt.show()

# Inspect texts with unusually high noun density (e.g., > 0.5)
high_noun_density = df[df['noun_density'] > 0.5]
print("Texts with very high noun density:")
print(high_noun_density[['essay_id', 'text', 'noun_density']])


