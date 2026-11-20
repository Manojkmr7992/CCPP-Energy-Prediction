from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "CCPP_data.csv"

FEATURES = ["AT", "V", "AP", "RH"]
TARGET = "PE"


def load_and_deduplicate(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    initial_rows = len(df)
    df_clean = df.drop_duplicates().reset_index(drop=True)
    removed = initial_rows - len(df_clean)
    print(f"Loaded {initial_rows:,} rows | Removed {removed} exact duplicates | Remaining: {len(df_clean):,}")
    return df_clean


def verify_zero_overlap(X_train: pd.DataFrame, X_test: pd.DataFrame) -> None:
    overlap = pd.merge(X_train, X_test, on=FEATURES, how="inner")
    overlap_count = len(overlap)
    print(f"Overlapping feature combinations between train and test: {overlap_count}")
    if overlap_count > 0:
        raise AssertionError(f"Data leakage detected: {overlap_count} duplicate feature rows found across splits.")


def main() -> None:
    df = load_and_deduplicate(DATA_PATH)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    print(f"\n=== Split Dimensions ===")
    print(f"X_train: {X_train.shape} | y_train: {y_train.shape}")
    print(f"X_test:  {X_test.shape} | y_test:  {y_test.shape}")

    print("\n=== Leakage Verification ===")
    verify_zero_overlap(X_train, X_test)


if __name__ == "__main__":
    main()

'''
Loaded 9,568 rows | Removed 41 exact duplicates | Remaining: 9,527

=== Split Dimensions ===
X_train: (7621, 4) | y_train: (7621,)
X_test:  (1906, 4) | y_test:  (1906,)

=== Leakage Verification ===
Overlapping feature combinations between train and test: 0
'''


from itertools import combinations
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "CCPP_data.csv"

FEATURES = ["AT", "V", "AP", "RH"]
TARGET = "PE"


def load_and_deduplicate(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    initial_rows = len(df)
    df_clean = df.drop_duplicates().reset_index(drop=True)
    removed = initial_rows - len(df_clean)
    print(f"Loaded {initial_rows:,} rows | Removed {removed} exact duplicates | Remaining: {len(df_clean):,}")
    return df_clean


def split_train_val_test(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    # 1. Hold out 20% Test set (leaves 80% Train+Val)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=random_state,
    )

    # 2. Split 80% Train+Val into 60% Train and 20% Validation (0.25 * 0.80 = 0.20)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,
        random_state=random_state,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def verify_pairwise_zero_overlap(partitions: dict[str, pd.DataFrame]) -> None:
    for (name_a, df_a), (name_b, df_b) in combinations(partitions.items(), 2):
        overlap = pd.merge(df_a, df_b, on=FEATURES, how="inner")
        overlap_count = len(overlap)
        print(f"Overlap ({name_a} vs {name_b}): {overlap_count}")
        if overlap_count > 0:
            raise AssertionError(
                f"Data leakage detected: {overlap_count} duplicate rows between {name_a} and {name_b}."
            )


def main() -> None:
    df = load_and_deduplicate(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X, y, random_state=42)

    total_split_rows = len(X_train) + len(X_val) + len(X_test)
    print("\n=== Partition Dimensions (60 / 20 / 20) ===")
    print(f"Train:      {X_train.shape} ({len(X_train) / total_split_rows:.2%})")
    print(f"Validation: {X_val.shape} ({len(X_val) / total_split_rows:.2%})")
    print(f"Test:       {X_test.shape} ({len(X_test) / total_split_rows:.2%})")
    print(f"Total rows across all partitions: {total_split_rows:,}")

    assert total_split_rows == len(df), f"Expected {len(df)} total rows, got {total_split_rows}"

    print("\n=== Pairwise Leakage Verification ===")
    verify_pairwise_zero_overlap({
        "Train": X_train,
        "Validation": X_val,
        "Test": X_test,
    })


if __name__ == "__main__":
    main()
