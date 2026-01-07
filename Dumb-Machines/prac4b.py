# 1. IMPORT LIBRARIES
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 2. CREATE SAMPLE DATA (instead of CSV)
# Example dataset: simple iris-like data
data = {
    'sepal_length': [5.1, 4.9, 5.8, 6.0, 6.3, 5.0, 5.5, 6.7, 6.5, 5.7],
    'sepal_width':  [3.5, 3.0, 2.7, 3.4, 3.3, 3.6, 2.8, 3.1, 2.8, 2.5],
    'petal_length': [1.4, 1.4, 4.1, 4.5, 5.0, 1.5, 4.5, 5.6, 5.2, 4.0],
    'petal_width':  [0.2, 0.2, 1.3, 1.5, 2.0, 0.2, 1.5, 2.4, 2.0, 1.2],
    'species':      [ 'setosa','setosa','versicolor','versicolor',
                      'virginica', 'setosa','versicolor','virginica',
                      'virginica', 'versicolor'
                    ]
}
df = pd.DataFrame(data)

# 3. SPLIT FEATURES & TARGET
X = df.drop('species', axis=1)
y = df['species']

# Split into train & test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 4. BUILD KNN MODEL
k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# 5. PREDICTION
y_pred = knn.predict(X_test)

# 6. PRINT RESULTS
print("Test Samples:")
print(X_test)
print("\nActual Labels:")
print(y_test.values)
print("\nPredicted Labels:")
print(y_pred)

# Identify correct and wrong predictions
correct = y_test[y_test == y_pred]
wrong = y_test[y_test != y_pred]
print("\nCorrect Predictions:")
print(correct)
print("\nWrong Predictions:")
print(wrong)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy*100:.2f}%")
