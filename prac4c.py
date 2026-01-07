# 1. IMPORT LIBRARIES
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 2. CREATE SAMPLE DATA
# Simple dataset (like Iris)
data = {
    'sepal_length': [5.1, 4.9, 5.8, 6.0, 6.3, 5.0, 5.5, 6.7, 6.5, 5.7],
    'sepal_width':  [3.5, 3.0, 2.7, 3.4, 3.3, 3.6, 2.8, 3.1, 2.8, 2.5],
    'petal_length': [1.4, 1.4, 4.1, 4.5, 5.0, 1.5, 4.5, 5.6, 5.2, 4.0],
    'petal_width':  [0.2, 0.2, 1.3, 1.5, 2.0, 0.2, 1.5, 2.4, 2.0, 1.2],
    'species':      ['setosa','setosa','versicolor','versicolor',
                     'virginica','setosa','versicolor','virginica','virginica',
                     'versicolor']
}
df = pd.DataFrame(data)

# 3. SPLIT FEATURES & TARGET
X = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. BUILD DECISION TREE CLASSIFIER
# Control max_depth to avoid overfitting
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)

# 5. PREDICT & EVALUATE
y_pred = tree.predict(X_test)
print("Actual Labels:")
print(y_test.values)
print("\nPredicted Labels:")
print(y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy*100:.2f}%")

# 6. VISUALIZE THE TREE
plt.figure(figsize=(12,8))
plot_tree(tree, feature_names=X.columns, class_names=tree.classes_, filled=True,
rounded=True)
plt.title("Decision Tree Visualization")
plt.show()
