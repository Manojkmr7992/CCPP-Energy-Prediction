from pathlib import Path
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)

from step03_data_preparation import (
    DATA_PATH,
    FEATURES,
    PROJECT_ROOT,
    TARGET,
    load_and_deduplicate,
    split_train_val_test,
    verify_pairwise_zero_overlap,
)

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def evaluate_regression(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    return {
        "MAE (MW)": float(mean_absolute_error(y_true, y_pred)),
        "RMSE (MW)": float(root_mean_squared_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_deduplicate(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]

    # Reuse Step 03B 60/20/20 split (Test set remains quarantined)
    X_train, X_val, X_test, y_train, y_val, _ = split_train_val_test(X, y, random_state=42)
    verify_pairwise_zero_overlap({"Train": X_train, "Validation": X_val, "Test": X_test})

    # 1. Fit Linear Regression baseline for direct comparison
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_val_metrics = evaluate_regression(y_val, lr_model.predict(X_val))

    # 2. Fit Random Forest Regressor (initial untuned benchmark)
    rf_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)

    rf_train_metrics = evaluate_regression(y_train, rf_model.predict(X_train))
    rf_val_metrics = evaluate_regression(y_val, rf_model.predict(X_val))

    # 3. Build comparison table
    comparison_df = pd.DataFrame([
        {
            "Model": "Linear Regression (Baseline)",
            "Train MAE (MW)": evaluate_regression(y_train, lr_model.predict(X_train))["MAE (MW)"],
            "Train RMSE (MW)": evaluate_regression(y_train, lr_model.predict(X_train))["RMSE (MW)"],
            "Train R2": evaluate_regression(y_train, lr_model.predict(X_train))["R2"],
            "Val MAE (MW)": lr_val_metrics["MAE (MW)"],
            "Val RMSE (MW)": lr_val_metrics["RMSE (MW)"],
            "Val R2": lr_val_metrics["R2"],
        },
        {
            "Model": "Random Forest (n=100)",
            "Train MAE (MW)": rf_train_metrics["MAE (MW)"],
            "Train RMSE (MW)": rf_train_metrics["RMSE (MW)"],
            "Train R2": rf_train_metrics["R2"],
            "Val MAE (MW)": rf_val_metrics["MAE (MW)"],
            "Val RMSE (MW)": rf_val_metrics["RMSE (MW)"],
            "Val R2": rf_val_metrics["R2"],
        },
    ])

    print("\n=== Model Comparison (Training vs. Validation) ===")
    print(comparison_df.round(4).to_string(index=False))

    # 4. Feature importances sorted descending
    importance_df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": rf_model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    print("\n=== Random Forest Feature Importances ===")
    print(importance_df.round(4).to_string(index=False))

    # 5. Persist comparison CSV
    output_csv = ARTIFACTS_DIR / "step05_model_comparison.csv"
    comparison_df.round(6).to_csv(output_csv, index=False)
    print(f"\nSaved model comparison to: {output_csv.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
