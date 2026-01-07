import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error

# 1. GENERATE DATASET
X, y, coef_true = make_regression(
    n_samples=500,
    n_features=10,
    n_informative=5,  # 5 actual relevant features
    noise=20,
    coef=True,
    random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. FIT MODELS
ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.1)
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)
elastic.fit(X_train, y_train)

# 3. PREDICTIONS
pred_ridge = ridge.predict(X_test)
pred_lasso = lasso.predict(X_test)
pred_elastic = elastic.predict(X_test)

# 4. PERFORMANCE METRICS
mse_ridge = mean_squared_error(y_test, pred_ridge)
mse_lasso = mean_squared_error(y_test, pred_lasso)
mse_elastic = mean_squared_error(y_test, pred_elastic)
print("\n===== MODEL PERFORMANCE (MSE) =====")
print(f"Ridge
MSE: {mse_ridge:.3f}")
print(f"Lasso
MSE: {mse_lasso:.3f}")
print(f"ElasticNet MSE: {mse_elastic:.3f}")
# 5. COMPARE COEFFICIENTS
print("\n===== TRUE VS ESTIMATED COEFFICIENTS =====")
print("True Coefs:", coef_true)
print("Ridge Coefs:", ridge.coef_)
print("Lasso Coefs:", lasso.coef_)
print("ElasticNet Coefs:", elastic.coef_)

# 6. VISUALIZATION (Coefficient shrinkage)
plt.figure(figsize=(12,5))
plt.plot(coef_true, "ko-", label="True coefficients")
plt.plot(ridge.coef_, "bs-", label="Ridge")
plt.plot(lasso.coef_, "r^-", label="Lasso")
plt.plot(elastic.coef_, "g*-", label="ElasticNet")
plt.title("Coefficient Comparison: Ridge vs Lasso vs ElasticNet")
plt.xlabel("Feature Index")
plt.ylabel("Coefficient Magnitude")
plt.grid(True)
plt.legend()
plt.show()
