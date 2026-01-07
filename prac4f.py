# XGBoost Classifier on Iris Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target # Already numeric in sklearn dataset, but let's assume you had strings

# If y is string labels like 'setosa', 'versicolor', 'virginica'
# Example: uncomment below if using a CSV
# y = df['species'] # string labels
# le = LabelEncoder()
# y = le.fit_transform(y)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBoost classifier
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
warning  # avoid

# Train model
xgb_model.fit(X_train, y_train)

# Predict
y_pred = xgb_model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
# Optional: if you used LabelEncoder and want original string labels
# y_pred_labels = le.inverse_transform(y_pred)
# print(y_pred_labels)
