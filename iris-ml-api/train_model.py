from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

# Create the model
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X, y)

# Save the trained model
joblib.dump(model, "model.joblib")

print("Model trained and saved successfully!")