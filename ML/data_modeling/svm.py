"""
============================================================
  Thermoelectric Merit-ZT Prediction
  Model  : SUPPORT VECTOR MACHINE (SVR — RBF kernel)
  Dataset: dataset_of_thermoelectric_figures.csv
  Rows   : 32,635  |  Features : 118  |  Target : ZT
============================================================

  NOTE ON RUNTIME:
  SVM does NOT scale well to very large datasets.
  This script automatically trains on a 5,000-sample
  subset for speed. You can increase TRAIN_SAMPLE_SIZE
  below — but expect longer runtimes as you go above 10k.
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

DATA_PATH         = "/home/sudarshan/Documents/INFORMATICS/dataset/dataset/dataset_of_thermoelectric_figures.csv"  # <- update path if needed
TARGET_COL        = "ZT"
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
TRAIN_SAMPLE_SIZE =  26108 #increase if you have time (max = full dataset)

# ─────────────────────────────────────────────
#  STEP 1 — LOAD & CLEAN
# ─────────────────────────────────────────────

df = pd.read_csv(DATA_PATH)
print(f"Loaded  : {df.shape[0]} rows x {df.shape[1]} columns")

X = df.drop(columns=[TARGET_COL] + DROP_COLS)
X = X.select_dtypes(include=np.number)
y = df[TARGET_COL]

# Fill small missing values with column median
X = X.fillna(X.median(numeric_only=True))

feature_names = X.columns.tolist()
print(f"Features: {len(feature_names)}")
print(f"ZT range: [{y.min():.4f} , {y.max():.4f}]  |  mean = {y.mean():.4f}")

# ─────────────────────────────────────────────
#  STEP 2 — SPLIT & SCALE
#  NOTE: SVM is sensitive to scale — StandardScaler is mandatory
# ─────────────────────────────────────────────

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Sub-sample training set for SVM speed
if TRAIN_SAMPLE_SIZE < len(X_train_full):
    idx = np.random.RandomState(42).choice(
        len(X_train_full), size=TRAIN_SAMPLE_SIZE, replace=False
    )
    X_train = X_train_full.iloc[idx]
    y_train = y_train_full.iloc[idx]
    print(f"\nSVM training subset : {TRAIN_SAMPLE_SIZE} samples "
          f"(out of {len(X_train_full)} train rows)")
else:
    X_train = X_train_full
    y_train = y_train_full
    print(f"\nTraining on full set: {len(X_train)} samples")

print(f"Test  samples       : {len(y_test)}")

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─────────────────────────────────────────────
#  STEP 3 — TRAIN
# ─────────────────────────────────────────────

model = SVR(
    kernel="rbf",
    C=10,
    epsilon=0.05,
    gamma="scale"
)

print("\nRunning 5-fold cross-validation ...")
kf    = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2 = cross_val_score(model, X_train_sc, y_train, cv=kf, scoring="r2")

print("Training final SVM model ...")
model.fit(X_train_sc, y_train)
y_pred = model.predict(X_test_sc)

# ─────────────────────────────────────────────
#  STEP 4 — METRICS
# ─────────────────────────────────────────────

r2   = r2_score(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\n" + "="*48)
print("  SVM (RBF kernel) — RESULTS")
print("="*48)
print(f"  Training subset size   : {len(y_train)}")
print(f"  Support vectors        : {model.n_support_[0]}")
print(f"  CV  R2  (mean +/- std) : {cv_r2.mean():.4f} +/- {cv_r2.std():.4f}")
print(f"  Test R2                : {r2:.4f}")
print(f"  Test MSE               : {mse:.6f}")
print(f"  Test RMSE              : {rmse:.6f}")
print("="*48)

# ─────────────────────────────────────────────
#  STEP 5 — PLOTS  (Actual vs Predicted + Metrics)
# ─────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    f"SVM (RBF Kernel) — ZT Prediction  [trained on {len(y_train):,} samples]",
    fontsize=13, fontweight="bold"
)

# Actual vs Predicted
axes[0].scatter(y_test, y_pred, alpha=0.3, color="#C44E52",
                edgecolors="none", s=15)
lims = [min(float(y_test.min()), float(y_pred.min())),
        max(float(y_test.max()), float(y_pred.max()))]
axes[0].plot(lims, lims, "k--", lw=1.5, label="Ideal fit")
axes[0].set_xlabel("Actual ZT", fontsize=11)
axes[0].set_ylabel("Predicted ZT", fontsize=11)
axes[0].set_title(f"Actual vs Predicted\nR2 = {r2:.4f}", fontsize=11)
axes[0].legend(fontsize=9)

# Metric bar chart: R2, MSE, RMSE
metric_names = ["R2", "MSE", "RMSE"]
metric_vals  = [r2, mse, rmse]
bar_colors   = ["#C44E52", "#DD8452", "#4C72B0"]
bars = axes[1].bar(metric_names, metric_vals, color=bar_colors, alpha=0.85, width=0.4)
for bar, val in zip(bars, metric_vals):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max(metric_vals) * 0.01,
        f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold"
    )
axes[1].set_ylabel("Score / Error", fontsize=11)
axes[1].set_title("Performance Metrics\n(R2, MSE, RMSE)", fontsize=11)

plt.tight_layout()
plt.savefig("/home/sudarshan/Documents/INFORMATICS/ML/svm_ZT_results.png", dpi=150, bbox_inches="tight")
print("\nPlot saved -> svm_ZT_results.png")
plt.show()
