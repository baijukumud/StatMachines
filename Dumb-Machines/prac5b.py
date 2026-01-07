# 1. IMPORT LIBRARIES
import numpy as np
from hmmlearn import hmm

# 2. CREATE SYNTHETIC OBSERVATION DATA
# Suppose we have 2 hidden states and 3 possible observable symbols (0, 1, 2)
# Example: Weather (hidden state) -> Observations (activity)
# States: 0 = Rainy, 1 = Sunny
# Observations: 0 = Walk, 1 = Shop, 2 = Clean

# Define observation sequences (training data)
observations = np.array([[0, 1, 2, 1, 0, 1, 2, 1, 0, 2]]).T
# Shape (n_samples, 1)

# 3. INITIALIZE HMM
model = hmm.MultinomialHMM(n_components=2, n_iter=100, random_state=42)

# 4. FIT THE MODEL
model.fit(observations)

# 5. PRINT MODEL PARAMETERS
print("Transition matrix (hidden states probabilities):")
print(model.transmat_)

print("\nEmission matrix (observation probabilities):")
print(model.emissionprob_)

print("\nStart probabilities (initial state probabilities):")
print(model.startprob_)

# 6. PREDICT HIDDEN STATES
hidden_states = model.predict(observations)
print("\nPredicted Hidden States:")
print(hidden_states)

# 7. GENERATE A SEQUENCE FROM HMM
X, Z = model.sample(10) # Generate 10 observations and hidden states
print("\nGenerated Observation Sequence:")
print(X.ravel())
print("Corresponding Hidden States:")
print(Z)
