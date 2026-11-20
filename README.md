# Combined Cycle Power Plant (CCPP) Energy Prediction

End-to-end supervised machine learning regression pipeline that predicts net hourly electrical energy output (**PE**, in megawatts) of a Combined Cycle Power Plant from ambient environmental and operating sensor measurements.

---

## 1. Problem Statement

Accurate hourly power output forecasting enables combined-cycle power plant operators to optimize grid dispatch schedules, fuel allocation, and turbine operating margins. Given four hourly average sensor readings—**Ambient Temperature (`AT`)**, **Exhaust Vacuum (`V`)**, **Ambient Pressure (`AP`)**, and **Relative Humidity (`RH`)**—the objective is to predict continuous net electrical power generation (**`PE`**, in MW) while minimizing large dispatch errors (**RMSE**).

---

## 2. Dataset Overview

* **Source:** UCI Machine Learning Repository — Combined Cycle Power Plant Dataset (Tüfekci, 2014; Kaya et al., 2012).
* **Original Size:** 9,568 hourly average observations across 5 numerical columns (`float64`), with **0 missing values**.
* **Data Quality & Deduplication:**
  * Identified **41 exact duplicate rows** (0.43%) and **0 conflicting target outputs** for identical feature combinations. Exact duplicates were removed prior to splitting to prevent cross-partition leakage, leaving **9,527 unique observations**.
  * Identified **55 records with `RH > 100%`** (maximum `100.16%`), retained as physically plausible sensor calibration/rounding tolerances under saturated atmospheric conditions.

| Variable | Description | Unit | Min | Median | Mean | Max |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| `AT` | Ambient Temperature | °C | 1.81 | 20.35 | 19.65 | 37.11 |
| `V` | Exhaust Vacuum | cm Hg | 25.36 | 52.08 | 54.31 | 81.56 |
| `AP` | Ambient Pressure | mbar | 992.89 | 1012.94 | 1013.26 | 1033.30 |
| `RH` | Relative Humidity | % | 25.56 | 74.98 | 73.31 | 100.16 |
| `PE` | **Net Electrical Power Output (Target)** | **MW** | **420.26** | **451.55** | **454.37** | **495.76** |

---

## 3. Methodology

1. **Exploratory Data Analysis & Diagnostics (`step01`–`step02`):**
   * Confirmed strong negative linear associations between `PE` and `AT` (r = -0.9481) and `V` (r = -0.8698), alongside noticeable nonlinear curvature at high ambient temperatures and distinct operating clusters in exhaust vacuum.
2. **Leak-Free Partitioning (`step03`):**
   * Deduplicated dataset (9,527 rows) split reproducibly (`random_state=42`) into **60% Training (5,715 rows)**, **20% Validation (1,906 rows)**, and **20% Holdout Test (1,906 rows)**, verified for **0 exact feature overlap** across all partition pairs.
3. **Model Selection & Hyperparameter Experiments (`step04`–`step06`):**
   * Evaluated **Multiple Linear Regression**, **5 Random Forest configurations** (`RF-01` through `RF-05`), and **3 Gradient Boosting configurations** (`GB-01` through `GB-03`) strictly on the Validation partition using **RMSE (MW)** as the primary selection metric.
4. **Final Refit & Holdout Evaluation (`step07`–`step08`):**
   * Refitted the winning **RF-01** model (`n_estimators=100`, `max_depth=None`, `min_samples_leaf=1`, `random_state=42`) on the combined **80% Training + Validation set (7,621 rows)**, evaluated once on the untouched **20% Test set (1,906 rows)**, serialized the model with `joblib`, and exposed a validated CLI inference interface.

---

## 4. Validation Model Comparison

All candidate algorithms were trained on the 60% Training set (5,715 rows) and benchmarked on the 20% Validation set (1,906 rows):

| Rank | Model | Configuration | Val MAE (MW) | Val RMSE (MW) | Val R² | Train–Val R² Gap | Fit Time (s) |
| :---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| **1** | **Random Forest (RF-01)** | `n_estimators=100, max_depth=None, min_samples_leaf=1` | **2.4739** | **3.4882** | **0.9569** | 0.0374 | 0.27 |
| 2 | Gradient Boosting (GB-03) | `n_estimators=300, learning_rate=0.05, max_depth=4` | 2.7138 | 3.6441 | 0.9530 | 0.0149 | 1.95 |
| 3 | Linear Regression (Baseline) | Ordinary Least Squares (Default) | 3.5559 | 4.5085 | 0.9280 | 0.0005 | 0.003 |

* **Key Takeaway:** Random Forest (`RF-01`) reduced validation RMSE by **22.6%** compared to Linear Regression (`3.4882 MW` vs. `4.5085 MW`) and **4.3%** compared to tuned Gradient Boosting (`3.6441 MW`), capturing nonlinear thermodynamic interactions across ambient sensors.

---

## 5. Final Holdout Test Results

After refitting `RF-01` on the combined 80% Training + Validation partition (7,621 rows), single-pass evaluation on the untouched 20% Test partition (1,906 rows) achieved:

| Metric | Validation (60% Train) | Final Test (80% Train+Val) |
| :--- | ---: | ---: |
| **MAE (MW)** | 2.4739 | **2.4291** |
| **RMSE (MW)** | 3.4882 | **3.4597** |
| **R²** | 0.9569 | **0.9593** |

---

## 6. Project Structure & How to Run

### Directory Layout
```text
CCPP-Energy-Prediction/
├── data/
│   └── CCPP_data.csv
├── src/
│   ├── step01_data_exploration.py
│   ├── step02_exploratory_analysis.py
│   ├── step02_data_visualization.py
│   ├── step03_data_preparation.py
│   ├── step04_linear_regression.py
│   ├── step04_model_diagnostics.py
│   ├── step05_random_forest.py
│   ├── step05_rf_tuning.py
│   ├── step06_gradient_boosting.py
│   ├── step07_final_model.py
│   └── step08_predict.py
├── artifacts/
│   ├── diagnostics/
│   ├── models/
│   │   └── ccpp_random_forest.joblib
│   ├── visualizations/
│   ├── step05_model_comparison.csv
│   ├── step05_rf_tuning_results.csv
│   ├── step06_model_comparison.csv
│   └── step07_final_test_metrics.csv
└── README.md
```

### Environment Setup & Execution
```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install required dependencies
python -m pip install pandas numpy scikit-learn matplotlib seaborn joblib

# 3. Run EDA, model comparison, and final model training
python3 src/step01_data_exploration.py
python3 src/step02_exploratory_analysis.py
python3 src/step02_data_visualization.py
python3 src/step06_gradient_boosting.py
python3 src/step07_final_model.py

# 4. Run CLI inference (defaults: AT=20.0, V=50.0, AP=1013.0, RH=70.0)
python3 src/step08_predict.py
python3 src/step08_predict.py --at 25.0 --v 60.0 --ap 1012.0 --rh 65.0
```

*Example CLI Output (`AT=20.0°C, V=50.0 cm Hg, AP=1013.0 mbar, RH=70.0%`):*
```text
Predicted Net Electrical Output (PE): 456.44 MW
```

---

## 7. Limitations & Future Improvements

* **Absence of Temporal Ordering:** The UCI dataset contains shuffled hourly records without timestamps; therefore, random holdout evaluation measures interpolation accuracy rather than future-period forecasting under seasonal drift or equipment degradation.
* **Operating Envelope Controls:** The CLI interface enforces schema, numeric type, and finite-value validation, but does not yet block out-of-distribution sensor readings outside the physical training envelope (`AT: 1.81–37.11°C`, `V: 25.36–81.56 cm Hg`, `AP: 992.89–1033.30 mbar`, `RH: 25.56–100.16%`).
* **Serialization Security & Portability:** Model artifacts persisted via `joblib` must only be loaded from trusted sources and executed in matching Python / `scikit-learn` library environments.
