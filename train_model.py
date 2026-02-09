import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("Loading dataset...")
df = pd.read_csv("data/processed/merged_data.csv", low_memory=False)

print("Original Shape:", df.shape)

# Fix column names
df.columns = df.columns.str.strip()

# Replace infinite values
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Fill NaN with 0 instead of dropping rows
df.fillna(0, inplace=True)

# Convert Label
df['Label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

print("\nLabel Distribution:")
print(df['Label'].value_counts())

# Keep only numeric columns
df = df.select_dtypes(include=[np.number])

print("After numeric filtering:", df.shape)

# Balance dataset safely
normal = df[df['Label'] == 0]
attack = df[df['Label'] == 1]

min_count = min(len(normal), len(attack))

normal_sample = normal.sample(n=min_count, random_state=42)
attack_sample = attack.sample(n=min_count, random_state=42)

df_balanced = pd.concat([normal_sample, attack_sample])

print("\nBalanced Distribution:")
print(df_balanced['Label'].value_counts())

# Features & target
X = df_balanced.drop("Label", axis=1)
y = df_balanced["Label"]

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

print("\nTraining model...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/random_forest_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\nModel saved successfully.")
