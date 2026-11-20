from time import perf_counter
import joblib
from numpy.typing import ArrayLike
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
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
MODELS_DIR = ARTIFACTS_DIR / "models"
MODEL_PATH = MODELS_DIR / "ccpp_random_forest.joblib"
METRICS_PATH = ARTIFACTS_DIR / "step07_final_test_metrics.csv"


def evaluate_regression(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    return {
        "MAE (MW)": float(mean_absolute_error(y_true, y_pred)),
        "RMSE (MW)": float(root_mean_squared_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_deduplicate(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]

    # 1. Reproduce exact 60 / 20 / 20 split
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        X, y, random_state=42
    )
    verify_pairwise_zero_overlap({
        "Train": X_train,
        "Validation": X_val,
        "Test": X_test,
    })

    # 2. Combine Train (60%) + Validation (20%) into full 80% development set
    X_train_full = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_train_full = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)
    verify_pairwise_zero_overlap({
        "Combined Train+Val (80%)": X_train_full,
        "Test (20%)": X_test,
    })

    # 3. Fit frozen RF-01 model once on 80% combined dataset
    final_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )

    start = perf_counter()
    final_model.fit(X_train_full, y_train_full)
    fit_seconds = perf_counter() - start

    # 4. Evaluate once on untouched 20% Test set
    train_full_metrics = evaluate_regression(y_train_full, final_model.predict(X_train_full))
    test_preds = final_model.predict(X_test)
    test_metrics = evaluate_regression(y_test, test_preds)

    summary_df = pd.DataFrame([
        {
            "Model": "Random Forest (RF-01 Final)",
            "Train+Val Rows": len(X_train_full),
            "Test Rows": len(X_test),
            "Train+Val RMSE (MW)": train_full_metrics["RMSE (MW)"],
            "Test MAE (MW)": test_metrics["MAE (MW)"],
            "Test RMSE (MW)": test_metrics["RMSE (MW)"],
            "Test R2": test_metrics["R2"],
            "Fit Time (s)": fit_seconds,
        }
    ])

    print("\n=== Final Holdout Test Evaluation (RF-01) ===")
    print(summary_df.round(4).to_string(index=False))

    # Sample predictions for final presentation demo
    demo_df = X_test.head(5).copy().reset_index(drop=True)
    demo_df["Actual PE (MW)"] = y_test.head(5).to_numpy()
    demo_df["Predicted PE (MW)"] = test_preds[:5]
    demo_df["Residual (MW)"] = demo_df["Actual PE (MW)"] - demo_df["Predicted PE (MW)"]

    print("\n=== Sample Holdout Test Predictions ===")
    print(demo_df.round(4).to_string(index=False))

    # 5. Persist serialized model artifact and final metrics CSV
    joblib.dump(final_model, MODEL_PATH, compress=3)
    summary_df.round(6).to_csv(METRICS_PATH, index=False)

    print(f"\nSaved trained model artifact to: {MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Saved final test metrics to:     {METRICS_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

'''
Loaded 9,568 rows | Removed 41 exact duplicates | Remaining: 9,527
Overlap (Train vs Validation): 0
Overlap (Train vs Test): 0
Overlap (Validation vs Test): 0
Overlap (Combined Train+Val (80%) vs Test (20%)): 0

=== Final Holdout Test Evaluation (RF-01) ===
                      Model  Train+Val Rows  Test Rows  Train+Val RMSE (MW)  Test MAE (MW)  Test RMSE (MW)  Test R2  Fit Time (s)
Random Forest (RF-01 Final)            7621       1906               1.2398         2.4291          3.4597   0.9593        0.3772

=== Sample Holdout Test Predictions ===
   AT     V      AP    RH  Actual PE (MW)  Predicted PE (MW)  Residual (MW)
15.66 41.35 1001.68 86.26          463.57           464.5516        -0.9816
22.06 59.39 1015.26 85.04          441.31           442.8513        -1.5413
22.45 59.87 1018.92 72.73          445.34           444.3067         1.0333
15.11 40.71 1017.96 69.31          465.42           462.3140         3.1060
30.28 58.90 1005.03 54.59          439.06           436.8209         2.2391

Saved trained model artifact to: artifacts/models/ccpp_random_forest.joblib
Saved final test metrics to:     artifacts/step07_final_test_metrics.csv
'''