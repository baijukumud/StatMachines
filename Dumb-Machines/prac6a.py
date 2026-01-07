import numpy as np
import matplotlib.pyplot as plt

# Seed for reproducibility
np.random.seed(42)

# Create synthetic data
def generate_data(n_samples=10):
    X = np.linspace(0, 1, n_samples)
    y = 2 * X + 1 + 0.1 * np.random.randn(n_samples)
    return X.reshape(-1, 1), y

# Add bias term
def design_matrix(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])

# Bayesian Linear Regression Posterior
def compute_posterior(X, y, sigma2, prior_mean, prior_cov):
    XT_X = X.T @ X
    sigma_inv = np.linalg.inv(prior_cov)
    Sigma_n = np.linalg.inv(sigma_inv + XT_X / sigma2)
    mu_n = Sigma_n @ (sigma_inv @ prior_mean + X.T @ y / sigma2)
    return mu_n, Sigma_n

# Sample from posterior
def sample_posterior(mu, cov, n_samples=5):
    return np.random.multivariate_normal(mu, cov, n_samples)

# Generate data
X_raw, y = generate_data(n_samples=10)
X = design_matrix(X_raw)

# Prior: zero-mean Gaussian
prior_mean = np.zeros(X.shape[1])
prior_cov = np.eye(X.shape[1]) * 1.0

# Noise variance
sigma2 = 0.01

# Posterior
mu_n, Sigma_n = compute_posterior(X, y, sigma2, prior_mean, prior_cov)

# Plotting
x_plot = np.linspace(0, 1, 100).reshape(-1, 1)
X_plot = design_matrix(x_plot)

# Plot samples from prior
prior_samples = sample_posterior(prior_mean, prior_cov, n_samples=5)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for w in prior_samples:
    y_plot = X_plot @ w
    plt.plot(x_plot, y_plot, lw=1)
plt.title("Samples from Prior")
plt.xlabel("x")
plt.ylabel("y")

# Plot samples from posterior
posterior_samples = sample_posterior(mu_n, Sigma_n, n_samples=5)
plt.subplot(1, 2, 2)
for w in posterior_samples:
    y_plot = X_plot @ w
    plt.plot(x_plot, y_plot, lw=1)
plt.scatter(X_raw, y, c='red', label='Data')
plt.title("Samples from Posterior")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

plt.tight_layout()
plt.show()
