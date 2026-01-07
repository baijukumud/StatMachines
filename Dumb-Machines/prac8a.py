# Bayesian Learning in Python: Two Examples
# 1) Parameter inference with a Beta-Binomial model
# 2) Classification with Multinomial Naive Bayes (Dirichlet prior / Laplace smoothing)
# WITHOUT using pandas

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from collections import Counter
from math import log, exp

# --- Part 1: Beta-Binomial (learning the bias of a coin) ---

alpha_prior, beta_prior = 2, 2

# Prior: Beta(alpha, beta)
# Observed data: 1=heads, 0=tails
observations = np.array([1,1,0,1,0,1,1,1,0,1, 1,0,1,1,1,0,1,1,1,1])
n_heads = observations.sum()
n_tails = len(observations) - n_heads

# Posterior parameters
alpha_post = alpha_prior + n_heads
beta_post = beta_prior + n_tails

# Posterior summaries
posterior_mean = alpha_post / (alpha_post + beta_post)
posterior_map = (alpha_post - 1) / (alpha_post + beta_post - 2) if (alpha_post>1 and
beta_post>1) else np.nan
predictive_next_head = posterior_mean

print("=== Beta-Binomial Inference (Coin Bias) ===")
print(f"Prior Beta({alpha_prior}, {beta_prior})")
print(f"Observed: {len(observations)} flips => heads={n_heads}, tails={n_tails}")
print(f"Posterior Beta({alpha_post}, {beta_post})")
print(f"Posterior mean of p(heads): {posterior_mean:.4f}")
print(f"Posterior MAP of p(heads): {posterior_map:.4f}")
print(f"Posterior predictive P(next flip = heads): {predictive_next_head:.4f}\n")

# Plot prior and posterior densities
x = np.linspace(0, 1, 500)
plt.figure()
plt.plot(x, beta.pdf(x, alpha_prior, beta_prior), label=f"Prior
Beta({alpha_prior},{beta_prior})")
plt.plot(x, beta.pdf(x, alpha_post, beta_post), label=f"Posterior
Beta({alpha_post},{beta_post})")
plt.title("Prior vs Posterior over coin bias p")
plt.xlabel("p (probability of heads)")
plt.ylabel("Density")
plt.legend()
plt.show()

# --- Part 2: Multinomial Naive Bayes (Dirichlet prior) ---

train_docs = [
    ("spam", "win cash now"),
    ("spam", "limited offer claim prize"),
    ("spam", "win big prize cash"),
    ("ham", "meeting at noon"),
    ("ham", "let us schedule a meeting"),
    ("ham", "please review the report"),
    ("ham", "see you at noon"),
]

test_docs = [
    ("?", "win a prize now"),
    ("?", "see you at the meeting"),
    ("?", "claim your cash offer"),
    ("?", "review the schedule"),
]

def tokenize(s):
    return s.lower().split()

# Build vocabulary
classes = sorted(set(lbl for lbl, _ in train_docs))
vocab = sorted(set(w for _, text in train_docs for w in tokenize(text)))
word_to_idx = {w:i for i,w in enumerate(vocab)}

alpha = 1.0  # Dirichlet prior smoothing

# Count words per class
class_doc_counts = Counter(lbl for lbl, _ in train_docs)
class_token_counts = {c: np.zeros(len(vocab), dtype=np.int64) for c in classes}
class_total_tokens = {c: 0 for c in classes}

for lbl, text in train_docs:
    toks = tokenize(text)
    for t in toks:
        class_token_counts[lbl][word_to_idx[t]] += 1
        class_total_tokens[lbl] += 1

# Class prior with smoothing
alpha_class = 1.0
num_classes = len(classes)
total_docs = len(train_docs)
class_prior_logprob = {}
for c in classes:
    class_prior_logprob[c] = log(
        (class_doc_counts[c] + alpha_class) /
        (total_docs + num_classes * alpha_class)
    )

# Conditional probabilities P(w|c)
cond_prob = {}
