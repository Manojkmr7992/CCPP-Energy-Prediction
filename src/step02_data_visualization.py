from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for clean script execution
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "CCPP_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "visualizations"


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
    })


def save_and_close(fig: plt.Figure, filename: str) -> None:
    out_path = OUTPUT_DIR / filename
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path.relative_to(PROJECT_ROOT)}")


def plot_pe_histogram(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["PE"], bins=40, color="#2b5c8f", edgecolor="white", alpha=0.85)
    ax.axvline(df["PE"].mean(), color="#d95f02", linestyle="--", linewidth=1.5, label=f"Mean: {df['PE'].mean():.2f} MW")
    ax.axvline(df["PE"].median(), color="#1b9e77", linestyle=":", linewidth=1.8, label=f"Median: {df['PE'].median():.2f} MW")
    ax.set_title("Distribution of Net Hourly Electrical Energy Output (PE)")
    ax.set_xlabel("Electrical Power Output (PE) [MW]")
    ax.set_ylabel("Frequency (Hourly Observations)")
    ax.legend()
    save_and_close(fig, "01_pe_histogram.png")


def plot_at_vs_pe(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["AT"], df["PE"], alpha=0.25, s=12, color="#1f78b4", edgecolors="none")
    ax.set_title("Ambient Temperature (AT) vs. Electrical Power Output (PE)")
    ax.set_xlabel("Ambient Temperature (AT) [°C]")
    ax.set_ylabel("Electrical Power Output (PE) [MW]")
    save_and_close(fig, "02_at_vs_pe_scatter.png")


def plot_v_vs_pe(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["V"], df["PE"], alpha=0.25, s=12, color="#33a02c", edgecolors="none")
    ax.set_title("Exhaust Vacuum (V) vs. Electrical Power Output (PE)")
    ax.set_xlabel("Exhaust Vacuum (V) [cm Hg]")
    ax.set_ylabel("Electrical Power Output (PE) [MW]")
    save_and_close(fig, "03_v_vs_pe_scatter.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    corr = df.corr(method="pearson")
    sns.heatmap(
        corr,
        annot=True,
        fmt=".4f",
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Pearson Correlation (r)"},
        ax=ax,
    )
    ax.set_title("Pearson Correlation Heatmap — CCPP Variables")
    save_and_close(fig, "04_correlation_heatmap.png")


def plot_pe_boxplot(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(x=df["PE"], color="#6baed6", width=0.4, flierprops={"marker": "o", "markersize": 4}, ax=ax)
    ax.set_title("Spread and Outlier Inspection of Electrical Power Output (PE)")
    ax.set_xlabel("Electrical Power Output (PE) [MW]")
    save_and_close(fig, "05_pe_boxplot.png")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} observations from {DATA_PATH.name}\n")

    plot_pe_histogram(df)
    plot_at_vs_pe(df)
    plot_v_vs_pe(df)
    plot_correlation_heatmap(df)
    plot_pe_boxplot(df)


if __name__ == "__main__":
    main()
'''
=== Correlation Matrix ===
        AT       V      AP      RH      PE
AT  1.0000  0.8441 -0.5075 -0.5425 -0.9481
V   0.8441  1.0000 -0.4135 -0.3122 -0.8698
AP -0.5075 -0.4135  1.0000  0.0996  0.5184
RH -0.5425 -0.3122  0.0996  1.0000  0.3898
PE -0.9481 -0.8698  0.5184  0.3898  1.0000

=== Feature Correlations with PE ===
AT   -0.9481
V    -0.8698
RH    0.3898
AP    0.5184

=== Strongest Correlated Feature ===
Feature: AT
Correlation: -0.9481

=== Step 02B: Data Quality Investigation ===
1. Rows with RH > 100%: 55
2. Repeated feature combinations (AT, V, AP, RH): 41
3. Exact duplicate rows: 41 (0.4285%)
4. Input combinations with >1 distinct PE value: 0
Matplotlib is building the font cache; this may take a moment.
Loaded 9,568 observations from CCPP_data.csv

Saved: artifacts/visualizations/01_pe_histogram.png
Saved: artifacts/visualizations/02_at_vs_pe_scatter.png
Saved: artifacts/visualizations/03_v_vs_pe_scatter.png
Saved: artifacts/visualizations/04_correlation_heatmap.png
Saved: artifacts/visualizations/05_pe_boxplot.png

'''