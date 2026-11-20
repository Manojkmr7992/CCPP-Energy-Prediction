from time import perf_counter
from numpy.typing import ArrayLike
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
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

GB_CONFIGS = [
    {"Experiment": "GB-01", "n_estimators": 100, "learning_rate": 0.10, "max_depth": 3},
    {"Experiment": "GB-02", "n_estimators": 200, "learning_rate": 0.05, "max_depth": 3},
    {"Experiment": "GB-03", "n_estimators": 300, "learning_rate": 0.05, "max_depth": 4},
]


def evaluate_regression(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(root_mean_squared_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def benchmark_model(
    name: str,
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: str = "Default",
) -> dict[str, object]:
    start = perf_counter()
    model.fit(X_train, y_train)
    train_seconds = perf_counter() - start

    train_metrics = evaluate_regression(y_train, model.predict(X_train))
    val_metrics = evaluate_regression(y_val, model.predict(X_val))

    return {
        "Model": name,
        "Configuration": params,
        "Train MAE (MW)": train_metrics["MAE"],
        "Train RMSE (MW)": train_metrics["RMSE"],
        "Train R2": train_metrics["R2"],
        "Val MAE (MW)": val_metrics["MAE"],
        "Val RMSE (MW)": val_metrics["RMSE"],
        "Val R2": val_metrics["R2"],
        "R2 Gap": train_metrics["R2"] - val_metrics["R2"],
        "Train Time (s)": train_seconds,
    }


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_deduplicate(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]

    # Reuse Step 03B 60/20/20 split (Test set remains quarantined)
    X_train, X_val, X_test, y_train, y_val, _ = split_train_val_test(X, y, random_state=42)
    verify_pairwise_zero_overlap({"Train": X_train, "Validation": X_val, "Test": X_test})

    # 1. Evaluate all three Gradient Boosting configurations
    gb_rows = []
    for cfg in GB_CONFIGS:
        gb_model = GradientBoostingRegressor(
            n_estimators=cfg["n_estimators"],
            learning_rate=cfg["learning_rate"],
            max_depth=cfg["max_depth"],
            random_state=42,
        )
        param_str = f"n_estimators={cfg['n_estimators']}, lr={cfg['learning_rate']}, max_depth={cfg['max_depth']}"
        gb_rows.append(
            benchmark_model(cfg["Experiment"], gb_model, X_train, y_train, X_val, y_val, param_str)
        )

    gb_df = (
        pd.DataFrame(gb_rows)
        .sort_values(by=["Val RMSE (MW)", "Val MAE (MW)"], ascending=[True, True])
        .reset_index(drop=True)
    )
    gb_df.insert(0, "Rank", gb_df.index + 1)

    print("\n=== Gradient Boosting Experiment Results (Ranked by Val RMSE) ===")
    print(gb_df.round(4).to_string(index=False))

    # 2. Compare Best GB against Linear Regression and RF-01
    best_gb_row = gb_df.drop(columns=["Rank"]).iloc[0].to_dict()
    best_gb_row["Model"] = f"Gradient Boosting ({best_gb_row['Model']})"

    lr_row = benchmark_model(
        "Linear Regression (Baseline)",
        LinearRegression(),
        X_train,
        y_train,
        X_val,
        y_val,
    )

    rf_row = benchmark_model(
        "Random Forest (RF-01)",
        RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        X_train,
        y_train,
        X_val,
        y_val,
        "n_estimators=100, max_depth=None, min_samples_leaf=1",
    )

    overall_df = (
        pd.DataFrame([lr_row, rf_row, best_gb_row])
        .sort_values(by=["Val RMSE (MW)", "Val MAE (MW)"], ascending=[True, True])
        .reset_index(drop=True)
    )
    overall_df.insert(0, "Rank", overall_df.index + 1)

    print("\n=== Cross-Algorithm Validation Comparison ===")
    print(overall_df.round(4).to_string(index=False))

    output_csv = ARTIFACTS_DIR / "step06_model_comparison.csv"
    overall_df.round(6).to_csv(output_csv, index=False)
    print(f"\nSaved cross-algorithm comparison to: {output_csv.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()


'''
Overlap (Train vs Validation): 0
Overlap (Train vs Test): 0
Overlap (Validation vs Test): 0

=== Gradient Boosting Experiment Results (Ranked by Val RMSE) ===
 Rank Model                          Configuration  Train MAE (MW)  Train RMSE (MW)  Train R2  Val MAE (MW)  Val RMSE (MW)  Val R2  R2 Gap  Train Time (s)
    1 GB-03 n_estimators=300, lr=0.05, max_depth=4          2.3604           3.0631    0.9678        2.7138         3.6441  0.9530  0.0149          1.9513
    2 GB-02 n_estimators=200, lr=0.05, max_depth=3          2.7923           3.6341    0.9547        2.9532         3.9093  0.9459  0.0089          1.0409
    3 GB-01  n_estimators=100, lr=0.1, max_depth=3          2.7941           3.6450    0.9545        2.9568         3.9200  0.9456  0.0089          0.5935

=== Cross-Algorithm Validation Comparison ===
 Rank                        Model                                        Configuration  Train MAE (MW)  Train RMSE (MW)  Train R2  Val MAE (MW)  Val RMSE (MW)  Val R2  R2 Gap  Train Time (s)
    1        Random Forest (RF-01) n_estimators=100, max_depth=None, min_samples_leaf=1          0.9250           1.2922    0.9943        2.4739         3.4882  0.9569  0.0374          0.2737
    2    Gradient Boosting (GB-03)               n_estimators=300, lr=0.05, max_depth=4          2.3604           3.0631    0.9678        2.7138         3.6441  0.9530  0.0149          1.9513
    3 Linear Regression (Baseline)                                              Default          3.6531           4.5658    0.9285        3.5559         4.5085  0.9280  0.0005          0.0027

Saved cross-algorithm comparison to: artifacts/step06_model_comparison.csv
'''