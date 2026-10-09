import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

# 1. Load cleaned data
df = pd.read_csv("cleaned_churn.csv")

# 2. Separate inputs (X) from target (y)
X = df.drop(columns=["Churn"])
y = df["Churn"].map({"No": 0, "Yes": 1})

# 3. Identify categorical and numerical columns
categorical_cols = X.select_dtypes(
    include=["object", "str"]
).columns.tolist()

numerical_cols = X.select_dtypes(
    include=["number"]
).columns.tolist()

# 4. Prepare preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_cols,
        ),
        ("numerical", StandardScaler(), numerical_cols),
    ]
)

# 5. Split data before fitting preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# 6. Build the complete pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced",
            ),
        ),
    ]
)

# 7. Train
model.fit(X_train, y_train)

# 8. Evaluate on unseen test data
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("--- MODEL EVALUATION ---")
print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
print(f"ROC-AUC:  {roc_auc_score(y_test, probabilities):.3f}")

print("\n--- CLASSIFICATION REPORT ---")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["Stayed", "Churned"],
    )
)

print("--- CONFUSION MATRIX ---")
print(confusion_matrix(y_test, predictions))

# 9. Save the complete pipeline
joblib.dump(model, "churn_model.joblib")
print("\nSaved trained pipeline to churn_model.joblib")
