# Import Libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Binarizer

# Create Dataset
data = {
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Female'],
    'Age': [25, 30, 45, 35, 22],
    'Salary': [50000, 60000, 80000, 65000, 48000],
    'Purchased': [0, 1, 1, 0, 0]
}
df = pd.DataFrame(data)
print("Original Dataset:")
print(df)

# Label Encoding
le = LabelEncoder()
df['Gender_Encoded'] = le.fit_transform(df['Gender'])
print("\nAfter Label Encoding:")
print(df[['Gender', 'Gender_Encoded']])

# Feature Scaling (Standardization)
scaler = StandardScaler()
df[['Age_Scaled', 'Salary_Scaled']] = scaler.fit_transform(df[['Age', 'Salary']])
print("\nAfter Feature Scaling:")
print(df[['Age_Scaled', 'Salary_Scaled']])

# Binarization
binarizer = Binarizer(threshold=0.5)
df['Purchased_Binary'] = binarizer.fit_transform(df[['Purchased']])
print("\nAfter Binarization:")
print(df[['Purchased', 'Purchased_Binary']])
