# train_model.py
import os
import pandas as pd
import numpy as np # Not directly used in final code, but often helpful
import seaborn as sns # Not directly used in final code, but often helpful
import matplotlib.pyplot as plt # Not directly used in final code, but often helpful
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Data Loading and Initial Cleaning
df = pd.read_csv("data/data.csv", encoding='latin1')
df.columns = ['target', 'id', 'date', 'query', 'user', 'text']
df['sentiment'] = df['target'].map({0: 'negative', 2: 'neutral', 4: 'positive'})
df = df[['text', 'sentiment']] # Keep only needed columns
print("Original data loaded and columns processed. First 5 rows:")
print(df.head())

# 2. Text Cleaning Function and Application
def clean_text(text):
    text = str(text) # Ensure text is a string to prevent errors with non-string types
    text = re.sub(r'@\w+', '', text)  # Remove @username
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Apply cleaning to the text column
df['text'] = df['text'].apply(clean_text)
print("\nText cleaned. First 5 rows:")
print(df.head())


# 3. Label Encoding
label_encoder = LabelEncoder()
df['sentiment_encoded'] = label_encoder.fit_transform(df['sentiment'])
print("\nSentiment labels encoded.")

# 4. Feature and Target Split
X = df['text']
y = df['sentiment_encoded']

# 5. TF-IDF Vectorization
# Store the vectorizer to use it later for new data
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
X_vectorized = tfidf_vectorizer.fit_transform(X) # Fit and transform on the entire X

# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=22)

print("\n✅ TF-IDF conversion complete.")
print("🔢 Training features shape:", X_train.shape)
print("🎯 Training labels shape:", y_train.shape)

# 7. Logistic Regression Model Training and Evaluation
# It's good practice to assign trained models to specific names, e.g., lr_model, rf_model
lr_model = LogisticRegression(max_iter=1000, solver='liblinear') # Added max_iter and solver for robustness
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

print("\n✅ Logistic Regression Model trained.")
print("📊 LR Accuracy:", accuracy_score(y_test, y_pred_lr))
print("\n📋 LR Classification Report:\n", classification_report(y_test, y_pred_lr, target_names=label_encoder.classes_))
print("\n🌀 LR Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))

# 9. Saving the Trained Model, Vectorizer, and Label Encoder
# Use meaningful filenames.
# We'll save the Logistic Regression model as an example for the Flask app.
# You could save rf_model instead if it performs better.

joblib.dump(lr_model, 'logistic_regression_model.joblib')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.joblib')
joblib.dump(label_encoder, 'label_encoder.joblib')

print("\n✅ Logistic Regression Model saved as 'logistic_regression_model.joblib'")
print("✅ TF-IDF Vectorizer saved as 'tfidf_vectorizer.joblib'")
print("✅ Label Encoder saved as 'label_encoder.joblib'")