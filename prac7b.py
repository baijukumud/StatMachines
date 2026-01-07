import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from scipy.stats import uniform

# Load dataset
data = load_iris()
X, y = data.data, data.target

# Grid Search CV (fixed)
print("Grid Search CV:")
param_grid = [
    {'solver': ['lbfgs'], 'penalty': ['l2'], 'C': [0.01, 0.1, 1, 10, 100]},
    {'solver': ['saga'], 'penalty': ['l1', 'l2'], 'C': [0.01, 0.1, 1, 10, 100]}
]
grid = GridSearchCV(LogisticRegression(max_iter=500), param_grid, cv=5,
scoring='accuracy', n_jobs=1)
grid.fit(X, y)
print("Best Parameters (GridSearch):", grid.best_params_)
print("Best CV Score:", grid.best_score_)
print()

# Randomized Search CV (fixed)
print("Randomized Search CV:")
param_dist = {
    'C': uniform(loc=0.01, scale=10),
    'solver': ['saga', 'lbfgs'],
    # saga supports l1, l2; lbfgs only l2 (handled automatically)
    'penalty': ['l1', 'l2']
}
random_search = RandomizedSearchCV(LogisticRegression(max_iter=500),
param_distributions=param_dist, n_iter=20, cv=5, scoring='accuracy', random_state=42,
n_jobs=1)
random_search.fit(X, y)
print("Best Parameters (RandomizedSearch):", random_search.best_params_)
print("Best CV Score:",random_search.best_score_)
