from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)

from step03_data_preparation import (
    DATA_PATH,
    FEATURES,
    TARGET,
    load_and_deduplicate,
    split_train_val_test,
    verify_pairwise_zero_overlap,
)


def evaluate_regression(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    return {
        "MAE (MW)": mean_absolute_error(y_true, y_pred),
        "RMSE (MW)": root_mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def main() -> None:
    df = load_and_deduplicate(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]

    # Reuse Step 03B 60/20/20 split (Test partition remains quarantined)
    X_train, X_val, X_test, y_train, y_val, _ = split_train_val_test(X, y, random_state=42)
    verify_pairwise_zero_overlap({"Train": X_train, "Validation": X_val, "Test": X_test})

    # Fit Linear Regression strictly on the training set
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict and evaluate strictly on the validation set
    val_preds = model.predict(X_val)
    metrics = evaluate_regression(y_val, val_preds)

    print("\n=== Validation Metrics (Linear Regression Baseline) ===")
    for metric_name, value in metrics.items():
        print(f"{metric_name:<10}: {value:.4f}")

    # Inspect intercept and feature coefficients
    coef_df = pd.DataFrame({
        "Feature": FEATURES,
        "Coefficient": model.coef_,
    })
    print(f"\n=== Model Parameters ===\nIntercept: {model.intercept_:.4f}")
    print(coef_df.to_string(index=False))

    # Display first 10 actual vs predicted validation values
    comparison_df = pd.DataFrame({
        "Actual PE (MW)": y_val.iloc[:10].to_numpy(),
        "Predicted PE (MW)": val_preds[:10],
    })
    comparison_df["Error (MW)"] = comparison_df["Predicted PE (MW)"] - comparison_df["Actual PE (MW)"]

    print("\n=== First 10 Validation Predictions ===")
    print(comparison_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

'''
Loaded 9,568 rows | Removed 41 exact duplicates | Remaining: 9,527
Overlap (Train vs Validation): 0
Overlap (Train vs Test): 0
Overlap (Validation vs Test): 0

=== Validation Metrics (Linear Regression Baseline) ===
MAE (MW)  : 3.5559
RMSE (MW) : 4.5085
R2        : 0.9280

=== Model Parameters ===
Intercept: 451.1037
Feature  Coefficient
     AT    -1.967052
      V    -0.236990
     AP     0.065496
     RH    -0.158030

=== First 10 Validation Predictions ===
 Actual PE (MW)  Predicted PE (MW)  Error (MW)
         449.80           449.1412     -0.6588
         452.75           457.4275      4.6775
         469.83           467.5887     -2.2413
         443.55           447.4581      3.9081
         475.02           476.7924      1.7724
         456.57           456.6413      0.0713
         473.56           467.6016     -5.9584
         481.92           482.1199      0.1999
         470.74           465.4331     -5.3069
         429.84           435.5432      5.7032
'''