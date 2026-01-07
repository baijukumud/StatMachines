import math
from collections import defaultdict

class NaiveBayesClassifier:
    def _init_(self):
        self.class_probs = {}
        self.feature_probs = {}
        self.classes = set()

    def fit(self, X, y):
        """
        X: list of feature dictionaries
        y: list of labels
        """
        n = len(y)
        self.classes = set(y)
        
        # Count occurrences
        class_counts = defaultdict(int)
        feature_counts = {c: defaultdict(lambda: defaultdict(int)) for c in self.classes}
        
        for xi, label in zip(X, y):
            class_counts[label] += 1
            for feature, value in xi.items():
                feature_counts[label][feature][value] += 1
        
        # Compute class probabilities P(class)
        self.class_probs = {c: class_counts[c]/n for c in self.classes}
        
        # Compute conditional probabilities P(feature=value | class)
        self.feature_probs = {c: {} for c in self.classes}
        for c in self.classes:
            for feature in feature_counts[c]:
                total = sum(feature_counts[c][feature].values())
                self.feature_probs[c][feature] = {
                    val: (count/total) for val, count in feature_counts[c][feature].items()
                }

    def predict(self, x):
        """
        x: dictionary of feature:value
        returns: predicted class
        """
        class_scores = {}
        for c in self.classes:
            # Start with prior P(class)
            score = math.log(self.class_probs[c])
            for feature, value in x.items():
                if value in self.feature_probs[c].get(feature, {}):
                    score += math.log(self.feature_probs[c][feature][value])
                else:
                    # Handle unseen feature values with smoothing
                    score += math.log(1e-6)
            class_scores[c] = score
        
        # Return class with max score
        return max(class_scores, key=class_scores.get)

# ------------------------------
# Example dataset: Weather & Play
# ------------------------------
X = [
    {"outlook": "sunny", "temp": "hot", "humidity": "high", "wind": "weak"},
    {"outlook": "sunny", "temp": "hot", "humidity": "high", "wind": "strong"},
    {"outlook": "overcast", "temp": "hot", "humidity": "high", "wind": "weak"},
    {"outlook": "rain", "temp": "mild", "humidity": "high", "wind": "weak"},
    {"outlook": "rain", "temp": "cool", "humidity": "normal", "wind": "weak"},
    {"outlook": "rain", "temp": "cool", "humidity": "normal", "wind": "strong"},
    {"outlook": "overcast", "temp": "cool", "humidity": "normal", "wind": "strong"},
]

y = ["no", "no", "yes", "yes", "yes", "no", "yes"]

# Train model
nb = NaiveBayesClassifier()
nb.fit(X, y)

# Test sample
test_sample = {"outlook": "sunny", "temp": "cool", "humidity": "high", "wind": "strong"}
prediction = nb.predict(test_sample)

print("Test Sample:", test_sample)
print("Predicted Class:",prediction)
