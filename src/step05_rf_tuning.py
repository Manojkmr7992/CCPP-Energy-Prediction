from time import perf_counter
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

RF_CONFIGS = [
    {"Experiment": "RF-01", "n_estimators": 100, "max_depth": None, "min_samples_leaf": 1},
    {"Experiment": "RF-02", "n_estimators": 100, "max_depth": 15, "min_samples_leaf": 1},
    {"Experiment": "RF-03", "n_estimators": 100, "max_depth": 12, "min_samples_leaf": 2},
    {"Experiment": "RF-04", "n_estimators": 100, "max_depth": 10, "min_samples_leaf": 5},
    {"Experiment": "RF-05", "n_estimators": 200, "max_depth": 15, "min_samples_leaf": 2},
]


def evaluate_regression(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(root_mean_squared_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_deduplicate(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]

    # Reuse Step 03B 60/20/20 split (Test partition remains quarantined)
    X_train, X_val, X_test, y_train, y_val, _ = split_train_val_test(X, y, random_state=42)
    verify_pairwise_zero_overlap({"Train": X_train, "Validation": X_val, "Test": X_test})

    results = []

    for cfg in RF_CONFIGS:
        model = RandomForestRegressor(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            min_samples_leaf=cfg["min_samples_leaf"],
            random_state=42,
            n_jobs=-1,
        )

        start = perf_counter()
        model.fit(X_train, y_train)
        train_seconds = perf_counter() - start

        train_metrics = evaluate_regression(y_train, model.predict(X_train))
        val_metrics = evaluate_regression(y_val, model.predict(X_val))

        results.append({
            "Experiment": cfg["Experiment"],
            "n_estimators": cfg["n_estimators"],
            "max_depth": str(cfg["max_depth"]),
            "min_samples_leaf": cfg["min_samples_leaf"],
            "Train MAE (MW)": train_metrics["MAE"],
            "Train RMSE (MW)": train_metrics["RMSE"],
            "Train R2": train_metrics["R2"],
            "Val MAE (MW)": val_metrics["MAE"],
            "Val RMSE (MW)": val_metrics["RMSE"],
            "Val R2": val_metrics["R2"],
            "R2 Gap": train_metrics["R2"] - val_metrics["R2"],
            "Train Time (s)": train_seconds,
        })

    results_df = (
        pd.DataFrame(results)
        .sort_values(by=["Val RMSE (MW)", "Val MAE (MW)"], ascending=[True, True])
        .reset_index(drop=True)
    )
    results_df.insert(0, "Rank", results_df.index + 1)

    print("\n=== Random Forest Hyperparameter Tuning Results (Ranked by Val RMSE) ===")
    print(results_df.round(4).to_string(index=False))

    output_csv = ARTIFACTS_DIR / "step05_rf_tuning_results.csv"
    results_df.round(6).to_csv(output_csv, index=False)
    print(f"\nSaved tuning results to: {output_csv.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

'''
Loaded 9,568 rows | Removed 41 exact duplicates | Remaining: 9,527
Overlap (Train vs Validation): 0
Overlap (Train vs Test): 0
Overlap (Validation vs Test): 0

=== Random Forest Hyperparameter Tuning Results (Ranked by Val RMSE) ===
 Rank Experiment  n_estimators max_depth  min_samples_leaf  Train MAE (MW)  Train RMSE (MW)  Train R2  Val MAE (MW)  Val RMSE (MW)  Val R2  R2 Gap  Train Time (s)
    1      RF-01           100      None                 1          0.9250           1.2922    0.9943        2.4739         3.4882  0.9569  0.0374          0.3321
    2      RF-05           200        15                 2          1.2701           1.8232    0.9886        2.5080         3.5110  0.9563  0.0323          0.4048
    3      RF-02           100        15                 1          1.0724           1.4532    0.9928        2.4921         3.5124  0.9563  0.0365          0.2428
    4      RF-03           100        12                 2          1.5384           2.1300    0.9844        2.5652         3.5702  0.9549  0.0296          0.1987
    5      RF-04           100        10                 5          2.0794           2.8317    0.9725        2.6971         3.7019  0.9515  0.0211          0.1826

Saved tuning results to: artifacts/step05_rf_tuning_results.csv
'''