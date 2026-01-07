import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression

# Load dataset
data = load_iris()
X, y = data.data, data.target
# Model
model = LogisticRegression(max_iter=200)
# 1. K-Fold Cross Validation
print("K-Fold Cross Validation:")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores_kf = cross_val_score(model, X, y, cv=kf)
print("Scores:", scores_kf)
print("Mean Accuracy:", np.mean(scores_kf))
print()

# 2. Stratified K-Fold Cross Validation
print("Stratified K-Fold Cross Validation:")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_skf = cross_val_score(model, X, y, cv=skf)
print("Scores:", scores_skf)
print("Mean Accuracy:", np.mean(scores_skf))

# 3. Default cross_val_score with StratifiedKFold for classification
print("\nDefault 5-Fold (Stratified for classification):")
scores_default = cross_val_score(model, X, y, cv=5)
print("Scores:", scores_default)
print("Mean Accuracy:", np.mean(scores_default))
