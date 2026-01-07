import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 2. CREATE SAMPLE DATA
# Small dataset (like Iris)
data = {
    'sepal_length': [5.1, 4.9, 5.8, 6.0, 6.3, 5.0, 5.5, 6.7, 6.5, 5.7],
    'sepal_width':  [3.5, 3.0, 2.7, 3.4, 3.3, 3.6, 2.8, 3.1, 2.8, 2.5],
    'petal_length': [1.4, 1.4, 4.1, 4.5, 5.0, 1.5, 4.5, 5.6, 5.2, 4.0],
    'petal_width':  [0.2, 0.2, 1.3, 1.5, 2.0, 0.2, 1.5, 2.4, 2.0, 1.2],
    'species':      ['setosa','setosa','versicolor','versicolor','virginica',
                     'setosa','versicolor','virginica','virginica','versicolor'
                    ]
}
df = pd.DataFrame(data)

# 3. SPLIT FEATURES & TARGET
X = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. BUILD SVM MODEL
# Kernel can be 'linear', 'poly', 'rbf', 'sigmoid'
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train, y_train)

# 5. PREDICT & EVALUATE
y_pred = svm_model.predict(X_test)
print("Actual Labels:")
print(y_test.values)
print("\nPredicted Labels:")
print(y_pred)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy*100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=svm_model.classes_,
yticklabels=svm_model.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
