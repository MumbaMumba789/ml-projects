import pandas as pd

df = pd.read_csv("customer_churn.csv")

# Convert TotalCharges to numeric.
# Invalid or blank values become NaN.
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Remove the customer identifier.
df = df.drop(columns=["customerID"])

# Count missing values after conversion.
print("--- MISSING VALUES AFTER CLEANING ---")
print(df.isnull().sum())

# Remove rows with missing values.
df = df.dropna()

# Save the cleaned dataset.
df.to_csv("cleaned_churn.csv", index=False)

print("\n--- CLEANED DATASET ---")
print("Rows and columns:", df.shape)
print("\nChurn distribution:")
print(df["Churn"].value_counts())
print("\nSaved as cleaned_churn.csv")
