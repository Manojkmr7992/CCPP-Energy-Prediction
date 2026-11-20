import argparse
import math
from pathlib import Path
import joblib
import pandas as pd

from step03_data_preparation import FEATURES, PROJECT_ROOT

MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "ccpp_random_forest.joblib"


def validate_inputs(inputs: dict[str, float]) -> None:
    missing = set(FEATURES) - set(inputs.keys())
    if missing:
        raise ValueError(f"Missing required features: {sorted(missing)}")

    for feature in FEATURES:
        value = inputs[feature]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"Feature '{feature}' must be numeric, got {type(value).__name__}.")
        if not math.isfinite(float(value)):
            raise ValueError(f"Feature '{feature}' must be a finite number, got {value}.")


def predict_power_output(
    at: float,
    v: float,
    ap: float,
    rh: float,
    model_path: Path = MODEL_PATH,
) -> float:
    if not model_path.exists():
        raise FileNotFoundError(f"Saved model artifact not found at: {model_path}")

    raw_inputs = {"AT": at, "V": v, "AP": ap, "RH": rh}
    validate_inputs(raw_inputs)

    # Construct single-row DataFrame preserving exact training feature names and order
    input_df = pd.DataFrame([{feat: float(raw_inputs[feat]) for feat in FEATURES}], columns=FEATURES)

    model = joblib.load(model_path)
    prediction = float(model.predict(input_df)[0])
    return prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict Combined Cycle Power Plant net electrical output (PE in MW)."
    )
    parser.add_argument("--at", type=float, default=20.0, help="Ambient Temperature in °C (default: 20.0)")
    parser.add_argument("--v", type=float, default=50.0, help="Exhaust Vacuum in cm Hg (default: 50.0)")
    parser.add_argument("--ap", type=float, default=1013.0, help="Ambient Pressure in mbar (default: 1013.0)")
    parser.add_argument("--rh", type=float, default=70.0, help="Relative Humidity in %% (default: 70.0)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predicted_pe = predict_power_output(
        at=args.at,
        v=args.v,
        ap=args.ap,
        rh=args.rh,
    )

    print("=== CCPP Electrical Power Output Prediction ===")
    print(f"Inputs -> AT: {args.at:.2f} °C | V: {args.v:.2f} cm Hg | AP: {args.ap:.2f} mbar | RH: {args.rh:.2f} %")
    print(f"Predicted Net Electrical Output (PE): {predicted_pe:.2f} MW")


if __name__ == "__main__":
    main()
