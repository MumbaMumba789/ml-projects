# Used Car Selling Price Predictor

Predicts a used car's selling price (in lakhs) from its age, mileage, fuel
type, seller type, transmission, and ownership history, using polynomial
regression with the degree chosen by cross-validation, compared against a
Ridge-regularized version.

## Dataset

`car_data.csv` — 301 used car listings with columns: `Car_Name`, `Year`,
`Selling_Price`, `Present_Price`, `Driven_kms`, `Fuel_Type`, `Selling_type`,
`Transmission`, `Owner`.

## Project structure

```
.
├── car_data.csv            # dataset
├── car_price_model.ipynb    # exploratory notebook, walks through the full workflow
├── train.py                 # standalone script version — same logic, script-friendly
└── requirements.txt
```

## Usage

```bash
pip install -r requirements.txt
python train.py
```

This runs cross-validation to pick the polynomial degree, trains both a
Linear and a Ridge regression on that degree, prints RMSE/R²/MAE for each,
and saves the better-performing model to `car_price_model.joblib`.

## Approach

- `Year` is converted to `Car_Age` (age generalizes better than an absolute
  year, and doesn't need a new value each year)
- Categorical features (`Fuel_Type`, `Selling_type`, `Transmission`) are
  one-hot encoded
- Polynomial degree (1–3) is chosen via 5-fold cross-validation on the
  training set only, scored by R²
- Both plain and Ridge-regularized (`alpha=1.0`) polynomial regression are
  trained at the selected degree; whichever scores higher R² on the held-out
  test set is saved

## Results

Cross-validation selects **degree 1** — the relationship between these
features and price is close to linear once age is used instead of raw year,
and higher-degree polynomials overfit badly on this small (301-row) dataset.
Both models land around **R² ≈ 0.59** on the test set, with Ridge edging out
plain linear regression slightly.

## Notes / possible extensions

- `Car_Name` and `Present_Price` were left out of the current feature set —
  `Present_Price` in particular is a strong predictor (it's the car's current
  ex-showroom price) and would likely raise R² substantially if included.
- With more data, a tree-based model (Random Forest / Gradient Boosting)
  would likely outperform polynomial regression here.
