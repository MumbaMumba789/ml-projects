# Iris Species Classifier

A Decision Tree classifier that predicts an iris flower's species
(*setosa*, *versicolor*, or *virginica*) from four measurements: sepal
length, sepal width, petal length, and petal width.

## Dataset

`Iris.csv` — the classic [Iris flower dataset](https://archive.ics.uci.edu/dataset/53/iris)
(150 rows, 4 numeric features, 1 categorical target).

## Project structure

```
.
├── Iris.csv              # dataset
├── iris_model.ipynb       # exploratory notebook, walks through the full workflow
├── train.py               # standalone script version — same logic, script-friendly
└── requirements.txt
```

## Usage

```bash
pip install -r requirements.txt
python train.py
```

This prints an accuracy score, a classification report, and a confusion
matrix, then saves the fitted model to `iris_model.joblib`.

## Approach

- 80/20 stratified train/test split (stratified so all three species stay
  proportionally represented in both sets)
- `DecisionTreeClassifier` from scikit-learn, `random_state=42` for
  reproducibility
- Evaluated with accuracy, precision/recall/F1 per class, and a confusion
  matrix

## Results

On the held-out test set: **~93% accuracy**, with all misclassifications
occurring between *versicolor* and *virginica* (the two species with the
most feature overlap) — *setosa* is separated perfectly.
