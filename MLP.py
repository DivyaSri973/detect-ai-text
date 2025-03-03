import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sentence_transformers import SentenceTransformer
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score



# Load dataset
# Assuming the dataset is loaded as a DataFrame called `df`
df = pd.read_csv('Dataset_with_new_features.csv')

# Step 1: Preprocessing
# Remove rows with NaN in relevant columns
df_cleaned = df.dropna()

# Identify rows where 'generated' == 'generated'
rows_to_drop = df[df['generated'] == 'generated'].index

# Drop these rows
dataset = df.drop(rows_to_drop)


print(f"Number of rows before cleaning: {df.shape[0]}")
print(f"Number of rows after cleaning: {df_cleaned.shape[0]}")
model = SentenceTransformer('all-MiniLM-L6-v2')  # This is a smaller, faster model
text_embeddings = model.encode(df_cleaned['text'].tolist(), show_progress_bar=True)

embeddings_df = pd.DataFrame(text_embeddings)
embeddings_df.columns = [f'embedding_{i}' for i in range(embeddings_df.shape[1])]
# print(embeddings_df.head())
print("Done with embeddings")
print(embeddings_df.shape)



other_features = df_cleaned.drop(columns=['essay_id', 'text', 'generated'])

# Separate features and target variable
X = pd.concat([other_features, embeddings_df], axis=1)  # Drop irrelevant columns
y = df_cleaned['generated']  # Target column
numerical_cols = other_features.columns
print("Numerical Cols: ", numerical_cols)


scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

# Split data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)  # 70% train, 30% temp
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # Split temp into 50% validation and 50% test

print(f"Training set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Step 2: Build and train the MLP
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    max_iter=100,
    random_state=42
)

print("Training the MLP classifier...")
mlp.fit(X_train, y_train)

# Step 3: Evaluate on Validation Set
y_val_pred = mlp.predict(X_val)
y_val_prob = mlp.predict_proba(X_val)[:, 1]  # Probabilities for class 1

val_accuracy = accuracy_score(y_val, y_val_pred)
val_precision = precision_score(y_val, y_val_pred, average='binary')
val_recall = recall_score(y_val, y_val_pred, average='binary')
val_roc_auc = roc_auc_score(y_val, y_val_prob)

print("\nValidation Metrics:")
print(f"Accuracy: {val_accuracy:.4f}")
print(f"Precision: {val_precision:.4f}")
print(f"Recall: {val_recall:.4f}")
print(f"ROC AUC Score: {val_roc_auc:.4f}")

# Step 4: Final Evaluation on Test Set
y_test_pred = mlp.predict(X_test)
y_test_prob = mlp.predict_proba(X_test)[:, 1]  # Probabilities for class 1

test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred, average='binary')
test_recall = recall_score(y_test, y_test_pred, average='binary')
test_roc_auc = roc_auc_score(y_test, y_test_prob)

print("\nTest Metrics:")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"ROC AUC Score: {test_roc_auc:.4f}")