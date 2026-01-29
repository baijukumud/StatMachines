from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer

data = load_iris()
X, y = data.data, data.target

model = RandomForestClassifier(random_state=42)

param_space = {
    'n_estimators': Integer(10, 200),     'max_depth': Integer(1, 20),
    'min_samples_split': Real(0.01, 0.3), 'min_samples_leaf': Integer(1, 10),
    'max_features': Real(0.1, 1.0),
}

opt = BayesSearchCV(
    estimator=model, search_spaces=param_space,
    n_iter=50,  # Number of parameter settings to try
    cv=5,  # Number of cross-validation folds
    n_jobs=-1,  # Use all processors
    random_state=42
)

opt.fit(X, y)
print("Best Parameters:", opt.best_params_)
print("Best Cross-Validation Score:", opt.best_score_)
