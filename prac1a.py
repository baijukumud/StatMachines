import pandas as pd

# Step 1: Load CSV
df = pd.read_csv("students_data.csv")
print("🔹 Original Data:\n", df)

# Step 2: Clean formatting
df['Name'] = df['Name'].str.strip()
df['Gender'] = df['Gender'].str.strip().str.lower()

# Step 3: Convert Age to numeric and handle missing
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['Age'] = df['Age'].fillna(df['Age'].mean())

# Step 4: Handle missing Marks
df['Marks'] = pd.to_numeric(df['Marks'], errors='coerce')
df['Marks'] = df['Marks'].fillna(df['Marks'].mean())

# Step 5: Convert Join Date to datetime
df['Join Date'] = pd.to_datetime(df['Join Date'], errors='coerce')

# Step 6: Remove outliers from 'Marks' using IQR
Q1 = df['Marks'].quantile(0.25)
Q3 = df['Marks'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df_cleaned = df[(df['Marks'] >= lower) & (df['Marks'] <= upper)].copy()  # important: use .copy()

# Step 7: Save cleaned data
df_cleaned.to_csv("cleaned_students_data.csv", index=False)

# Step 8: Print cleaned result
print("\n Cleaned Data:\n",df_cleaned)
