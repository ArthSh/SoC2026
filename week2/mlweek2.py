"""
Week 2 - Machine Learning Basics
Summer of Code 2026

This script demonstrates:
1. Loading a dataset
2. Data preprocessing
3. Train-Test Split
4. Model Training
5. Prediction
6. Model Evaluation
"""

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# ------------------------------
# Load Dataset
# ------------------------------

iris = load_iris()

X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

print("Dataset Shape:", X.shape)
print("\nFirst Five Samples:")
print(X.head())

# ------------------------------
# Train-Test Split
# ------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples :", len(X_test))

# ------------------------------
# Model Training
# ------------------------------

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

# ------------------------------
# Predictions
# ------------------------------

predictions = model.predict(X_test)

# ------------------------------
# Evaluation
# ------------------------------

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report")
print(classification_report(y_test, predictions))

# ------------------------------
# Feature Importance
# ------------------------------

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

print("\nFeature Importance")
print(importance.sort_values(by="Importance", ascending=False))
