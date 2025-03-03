from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from kerastuner.tuners import Hyperband

# Step 1: Data Preprocessing

# Load the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for all texts
dataset_em = dataset.copy()
embeddings = model.encode(dataset_em['text'].tolist(), show_progress_bar=True)
dataset_em['embedding_feature'] = list(embeddings)  # Store the entire embedding vector as a single feature

# Drop irrelevant columns
features = dataset_em.drop(columns=[
    'essay_id', 'text', 'generated', 'source', 'temp', 'word_count', 
    'average_word_length', "flesch_kincaid", "punctuation_count", 
    'noun_count', 'verb_count', 'adjective_count', 'adverb_count', 
    'pos_counts', 'pos_density', 'misspelled_word_count', 'ai_ratio'
])

# Extract target
y = dataset_em['generated'].values

# Scale numerical features
numerical_cols = features.columns.drop('embedding_feature')  # Exclude the embedding feature
scaler = StandardScaler()
features[numerical_cols] = scaler.fit_transform(features[numerical_cols])

# Combine scaled numerical features and embeddings
X = np.hstack([features[numerical_cols].values, np.vstack(features['embedding_feature'])])

# Split into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Training set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Step 2: Hyperparameter Tuning with Validation

def build_model(hp):
    """Build an MLP model with tunable hyperparameters."""
    model = Sequential([
        Input(shape=(X.shape[1],)),  # Input shape based on combined feature vector
        Dense(hp.Int('units_1', 64, 256, step=32), activation='relu'),
        Dropout(hp.Float('dropout_1', 0.1, 0.5, step=0.1)),
        Dense(hp.Int('units_2', 32, 128, step=16), activation='relu'),
        Dropout(hp.Float('dropout_2', 0.1, 0.5, step=0.1)),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    model.compile(
        optimizer=Adam(learning_rate=hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# Use KerasTuner for hyperparameter tuning
tuner = Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=20,
    factor=3,
    directory='mlp_tuning',
    project_name='mlp_tune_embeddings'
)

# Perform the search with validation data
tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32)

# Retrieve the best model
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Best hyperparameters: {best_hps.values}")

best_model = tuner.hypermodel.build(best_hps)
history = best_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32)

# Step 3: Evaluate the Model
loss, accuracy = best_model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")
