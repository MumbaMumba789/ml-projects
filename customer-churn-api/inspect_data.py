import pandas as pd

df = pd.read_csv("customer_churn.csv")

print("\n--- FIRST 5 CUSTOMERS ---")
print(df.head())

print("\n--- DATASET SHAPE ---")
print(df.shape)

print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

print("\n--- DATA TYPES AND MISSING VALUES ---")
df.info()

print("\n--- CHURN DISTRIBUTION ---")
print(df["Churn"].value_counts())

print("\n--- MISSING VALUES PER COLUMN ---")
print(df.isnull().sum())
