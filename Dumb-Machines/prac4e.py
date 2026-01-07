import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 2. CREATE SAMPLE DATA
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

2# 3. SPLIT FEATURES & TARGET
X = df.drop('species', axis=1)
y = df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. SINGLE DECISION TREE
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"Decision Tree Accuracy: {accuracy_dt*100:.2f}%")

# 5. RANDOM FOREST ENSEMBLE
# Experiment with number of trees and feature sampling
n_trees_list = [5, 10, 50]
max_features_list = [1, 2, 3, None]

for n_trees in n_trees_list:
    for max_feat in max_features_list:
        rf = RandomForestClassifier(
            n_estimators=n_trees, max_features=max_feat, random_state=42
        )
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        print(
            f"Random Forest (trees={n_trees}", f"max_features={max_feat})", 
            f"Accuracy: {accuracy_rf*100:.2f}%")

# 6. COMPARISON
plt.figure(figsize=(6,4))
plt.bar(
    ['Decision Tree', 'RF 5', 'RF 10', 'RF 50'],
    [
        accuracy_dt,
        accuracy_score(
            y_test, RandomForestClassifier(n_estimators=5, random_state=42
        ).fit(X_train, y_train).predict(X_test)),
        accuracy_score(
            y_test, RandomForestClassifier(n_estimators=10, random_state=42
        ).fit(X_train, y_train).predict(X_test)),
        accuracy_score(
            y_test, RandomForestClassifier(n_estimators=50, random_state=42
        ).fit(X_train, y_train).predict(X_test))
    ]
)
plt.ylabel("Accuracy")
plt.title("Decision Tree vs Random Forest")
plt.show()
