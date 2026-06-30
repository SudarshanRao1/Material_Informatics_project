"""
============================================================
  Thermoelectric Merit-ZT Prediction
  Model  : GRADIENT BOOSTING MACHINE (GBM)
  Dataset: dataset_of_thermoelectric_figures.csv
  Rows   : 32,635  |  Features : 118  |  Target : ZT
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error, r2_score

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

DATA_PATH    = "/home/sudarshan/Documents/INFORMATICS/dataset/dataset/dataset_of_thermoelectric_figures.csv"   # <- update path if needed
TARGET_COL   = "ZT"
DROP_COLS = [

    # ========================================================
    # NON-NUMERIC
    # ========================================================

    'composition',

    # ========================================================
    # DIRECT TARGET LEAKAGE
    # ========================================================

    'Seebeck coefficient',

    'Thermal conductivity',

    'Electrical conductivity',

    'reduced_ZT_T',

    'ZT_numerator_proxy',

    'power_factor_proxy',

    'sigma_over_kappa',

    'kappa_electronic_WF',

    'kappa_lattice_proxy',

    # ========================================================
    # DUPLICATE FEATURES
    # ========================================================

    'range_X',

    'range_atomic_mass',

    'range_Z',

    'nonmetal_fraction',

    'range_atomic_radius',

    'total_valence_electrons_per_atom',

    'max_electronegativity',

    'min_electronegativity',

    'light_element_fraction',

    'element_diversity_index',

    'lanthanide_fraction',

    'max_atomic_mass',

    'std_Z',

    'std_X',

    'debye_temperature_estimate',

    # ========================================================
    # LOW INFORMATION
    # ========================================================

    'Seebeck_sign',

    # ========================================================
    # REDUNDANT LOG FEATURES
    # ========================================================

    'log_electrical_conductivity',

    'log_thermal_conductivity',

    # ========================================================
    # TEMPERATURE TRANSFORMS
    # ========================================================

    'T_squared',

    'log_T',

    'T_times_avg_Z',

    'T_times_VEC',

    'T_over_Debye',

    'T_over_avg_melting_point',

    # ========================================================
    # REDUNDANT ABS FEATURES
    # ========================================================

    'abs_Seebeck',

    'Seebeck_squared'
]
TRAIN_SAMPLE = 26108   # <- increase to len(X_train_full) on your own machine

# ─────────────────────────────────────────────
#  STEP 1 — LOAD & CLEAN
# ─────────────────────────────────────────────

df = pd.read_csv(DATA_PATH)
print(f"Loaded  : {df.shape[0]} rows x {df.shape[1]} columns")

X = df.drop(columns=[TARGET_COL] + DROP_COLS)
X = X.select_dtypes(include=np.number)
y = df[TARGET_COL]

X = X.fillna(X.median(numeric_only=True))

feature_names = X.columns.tolist()
print(f"Features: {len(feature_names)}")
print(f"ZT range: [{y.min():.4f} , {y.max():.4f}]  |  mean = {y.mean():.4f}")

# ─────────────────────────────────────────────
#  STEP 2 — SPLIT
#  NOTE: GBM does NOT need feature scaling
# ─────────────────────────────────────────────

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train, y_train = resample(X_train_full, y_train_full,
                             n_samples=TRAIN_SAMPLE, random_state=42)

print(f"\nTrain samples : {len(y_train)}  (set TRAIN_SAMPLE = {len(X_train_full)} on your machine for full data)")
print(f"Test  samples : {len(y_test)}")

# ─────────────────────────────────────────────
#  STEP 3 — TRAIN
# ─────────────────────────────────────────────

model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=4,
    min_samples_leaf=2,
    subsample=0.8,
    random_state=42
)

print("\nRunning 5-fold cross-validation ...")
kf    = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2 = cross_val_score(model, X_train, y_train, cv=kf, scoring="r2")

print("Training final GBM model ...")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ─────────────────────────────────────────────
#  STEP 4 — METRICS
# ─────────────────────────────────────────────

r2   = r2_score(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\n" + "="*48)
print("  GRADIENT BOOSTING MACHINE (GBM) — RESULTS")
print("="*48)
print(f"  CV  R2  (mean +/- std) : {cv_r2.mean():.4f} +/- {cv_r2.std():.4f}")
print(f"  Test R2                : {r2:.4f}")
print(f"  Test MSE               : {mse:.6f}")
print(f"  Test RMSE              : {rmse:.6f}")
print("="*48)

# Top 15 Feature Importances
imp_series = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
print("\n  Top 15 Feature Importances:")
print(imp_series.head(15).to_string())

# ─────────────────────────────────────────────
#  STEP 5 — PLOTS
# ─────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(19, 5))
fig.suptitle("Gradient Boosting (GBM) — ZT Prediction", fontsize=14, fontweight="bold")

# Actual vs Predicted
axes[0].scatter(y_test, y_pred, alpha=0.3, color="#2E86AB",
                edgecolors="none", s=15)
lims = [min(float(y_test.min()), float(y_pred.min())),
        max(float(y_test.max()), float(y_pred.max()))]
axes[0].plot(lims, lims, "r--", lw=1.5, label="Ideal fit")
axes[0].set_xlabel("Actual ZT", fontsize=11)
axes[0].set_ylabel("Predicted ZT", fontsize=11)
axes[0].set_title(f"Actual vs Predicted\nR2 = {r2:.4f}", fontsize=11)
axes[0].legend(fontsize=9)

# Metric bar chart: R2, MSE, RMSE
metric_names = ["R2", "MSE", "RMSE"]
metric_vals  = [r2, mse, rmse]
bar_colors   = ["#2E86AB", "#DD8452", "#55A868"]
bars = axes[1].bar(metric_names, metric_vals, color=bar_colors, alpha=0.85, width=0.4)
for bar, val in zip(bars, metric_vals):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max(metric_vals) * 0.01,
        f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold"
    )
axes[1].set_ylabel("Score / Error", fontsize=11)
axes[1].set_title("Performance Metrics\n(R2, MSE, RMSE)", fontsize=11)

# Top 15 Feature Importances
top15 = imp_series.head(15)
axes[2].barh(top15.index[::-1], top15.values[::-1], color="#2E86AB", alpha=0.85)
axes[2].set_xlabel("Importance", fontsize=11)
axes[2].set_title("Top 15 Feature Importances", fontsize=11)

plt.tight_layout()
plt.savefig("/home/sudarshan/Documents/INFORMATICS/ML/ML_tree_algos/gbm_ZT_results.png", dpi=150, bbox_inches="tight")
print("\nPlot saved -> gbm_ZT_results.png")
plt.show()
