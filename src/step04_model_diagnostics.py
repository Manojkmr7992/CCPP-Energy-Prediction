from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import seaborn as sns
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

DIAGNOSTICS_DIR = PROJECT_ROOT / "artifacts" / "diagnostics"


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
    })


def evaluate_regression(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    return {
        "MAE (MW)": float(mean_absolute_error(y_true, y_pred)),
        "RMSE (MW)": float(root_mean_squared_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def save_and_close(fig: plt.Figure, filename: str) -> None:
    out_path = DIAGNOSTICS_DIR / filename
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.relative_to(PROJECT_ROOT)}")


def plot_actual_vs_predicted(y_val: pd.Series, val_preds: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_val, val_preds, alpha=0.25, s=14, color="#1f78b4", edgecolors="none")

    min_val = min(float(y_val.min()), float(val_preds.min()))
    max_val = max(float(y_val.max()), float(val_preds.max()))
    ax.plot([min_val, max_val], [min_val, max_val], color="#d95f02", linestyle="--", linewidth=1.8, label="Perfect Fit (y = x)")

    ax.set_title("Validation: Actual vs. Predicted Electrical Output (PE)")
    ax.set_xlabel("Actual PE [MW]")
    ax.set_ylabel("Predicted PE [MW]")
    ax.legend()
    save_and_close(fig, "01_actual_vs_predicted.png")


def plot_residuals_vs_predicted(val_preds: np.ndarray, residuals: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(val_preds, residuals, alpha=0.25, s=14, color="#33a02c", edgecolors="none")
    ax.axhline(y=0.0, color="#d95f02", linestyle="--", linewidth=1.8, label="Zero Residual Baseline")

    ax.set_title("Validation: Residuals vs. Predicted PE")
    ax.set_xlabel("Predicted PE [MW]")
    ax.set_ylabel("Residual (Actual - Predicted) [MW]")
    ax.legend()
    save_and_close(fig, "02_residuals_vs_predicted.png")


def plot_residual_distribution(residuals: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(residuals, bins=40, kde=True, color="#2b5c8f", edgecolor="white", ax=ax)
    ax.axvline(0.0, color="#d95f02", linestyle="--", linewidth=1.8, label="Zero Error")
    ax.axvline(float(residuals.mean()), color="#1b9e77", linestyle=":", linewidth=1.8, label=f"Mean Residual: {residuals.mean():.4f} MW")

    ax.set_title("Validation: Distribution of Prediction Residuals")
    ax.set_xlabel("Residual (Actual - Predicted) [MW]")
    ax.set_ylabel("Frequency")
    ax.legend()
    save_and_close(fig, "03_residual_distribution.png")


def main() -> None:
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()

    df = load_and_deduplicate(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]

    # Reuse Step 03B 60/20/20 split (Test set remains untouched)
    X_train, X_val, X_test, y_train, y_val, _ = split_train_val_test(X, y, random_state=42)
    verify_pairwise_zero_overlap({"Train": X_train, "Validation": X_val, "Test": X_test})

    # Fit baseline Linear Regression strictly on training data
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 1 & 2. Compare Training vs. Validation metrics
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)

    metrics_df = pd.DataFrame({
        "Partition": ["Training (60%)", "Validation (20%)"],
        **{
            metric: [
                evaluate_regression(y_train, train_preds)[metric],
                evaluate_regression(y_val, val_preds)[metric],
            ]
            for metric in ["MAE (MW)", "RMSE (MW)", "R2"]
        },
    })
    print("\n=== Training vs. Validation Performance (Linear Regression) ===")
    print(metrics_df.round(4).to_string(index=False))

    # 3. Compute validation residuals (Actual - Predicted)
    residuals = y_val - val_preds

    # 4, 5 & 6. Generate and save diagnostic plots
    print("\n=== Generating Diagnostic Plots ===")
    plot_actual_vs_predicted(y_val, val_preds)
    plot_residuals_vs_predicted(val_preds, residuals)
    plot_residual_distribution(residuals)

    # 7. Identify the 10 largest absolute validation errors
    top_10_idx = residuals.abs().nlargest(10).index
    largest_errors_df = X_val.loc[top_10_idx].copy()
    largest_errors_df["Actual PE (MW)"] = y_val.loc[top_10_idx]
    largest_errors_df["Predicted PE (MW)"] = val_preds[y_val.index.get_indexer(top_10_idx)]
    largest_errors_df["Residual (MW)"] = residuals.loc[top_10_idx]
    largest_errors_df["Abs Error (MW)"] = residuals.loc[top_10_idx].abs()

    print("\n=== Top 10 Largest Absolute Validation Errors ===")
    print(largest_errors_df.round(4).to_string())


if __name__ == "__main__":
    main()

'''
Loaded 9,568 rows | Removed 41 exact duplicates | Remaining: 9,527
Overlap (Train vs Validation): 0
Overlap (Train vs Test): 0
Overlap (Validation vs Test): 0

=== Training vs. Validation Performance (Linear Regression) ===
       Partition  MAE (MW)  RMSE (MW)     R2
  Training (60%)    3.6531     4.5658 0.9285
Validation (20%)    3.5559     4.5085 0.9280

=== Generating Diagnostic Plots ===
Saved: artifacts/diagnostics/01_actual_vs_predicted.png
Saved: artifacts/diagnostics/02_residuals_vs_predicted.png
Saved: artifacts/diagnostics/03_residual_distribution.png

=== Top 10 Largest Absolute Validation Errors ===
         AT      V       AP     RH  Actual PE (MW)  Predicted PE (MW)  Residual (MW)  Abs Error (MW)
3113   8.74  36.30  1015.18  61.97          438.63           482.0056       -43.3756         43.3756
6770  16.84  56.03  1020.25  70.37          437.14           460.4013       -23.2613         23.2613
2932  16.51  51.86  1022.37  81.18          442.48           460.4692       -17.9892         17.9892
5854  20.58  54.50  1015.04  89.83          467.76           449.9906        17.7694         17.7694
202   21.19  42.48  1013.43  80.52          469.76           453.0051        16.7549         16.7549
6730  16.41  41.48  1016.28  45.19          451.93           468.4145       -16.4845         16.4845
6535   9.41  40.43  1025.68  79.49          492.88           477.6279        15.2521         15.2521
6636  22.97  42.48  1012.84  64.71          466.70           451.9636        14.7364         14.7364
4263  16.34  41.48  1015.92  48.24          453.60           468.0466       -14.4466         14.4466
8449   9.14  37.36  1015.22  78.06          491.97           478.4275        13.5425         13.5425
'''