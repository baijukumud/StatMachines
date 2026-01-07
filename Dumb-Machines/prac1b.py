# Load Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load & Display Dataset
iris = load_iris(as_frame=True)
df = iris.frame
print("First 5 rows of dataset:")
print(df.head())

# Descriptive Statistics
print("\nDescriptive Summary Statistics:")
print(df.describe())

# Identify Features & Target
X = df.drop('target', axis=1)  # Features
y = df['target']               # Target

print("\nFeatures:")
print(X.columns)

print("\nTarget Variable:")
print(y.name)

# Univariate Analysis (Matplotlib)
plt.figure()
plt.hist(df['sepal length (cm)'])
plt.title("Histogram of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.show()

# Bivariate Analysis (Matplotlib)
plt.figure()
plt.scatter(df['sepal length (cm)'], df['petal length (cm)'])
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.show()

# Univariate Analysis (Seaborn)
plt.figure()
sns.histplot(df['petal length (cm)'])
plt.title("Seaborn Histogram of Petal Length")
plt.show()

# Bivariate Analysis (Seaborn)
plt.figure()
sns.scatterplot(
    x='sepal length (cm)', y='petal length (cm)',
    hue='target', data=df
)
plt.title("Seaborn Scatter Plot with Target")
plt.show()
