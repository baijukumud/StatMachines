# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

# Step 1: Generate synthetic data
X, y_true = make_blobs(
    n_samples=500, centers=3, cluster_std=1.0, random_state=42)

# Step 2: Fit a Gaussian Mixture Model
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X)
labels = gmm.predict(X) # Cluster assignments

# Step 3: Plot the clusters
plt.figure(figsize=(8,6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=40)
plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='red', marker='x', s=100,
label='Centroids')
plt.title("Gaussian Mixture Model Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

# Step 4: Display probabilities for each sample (density estimation)
probabilities = gmm.predict_proba(X)
print("First 5 samples probabilities:\n", probabilities[:5])
