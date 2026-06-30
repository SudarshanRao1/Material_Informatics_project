# ============================================================
# LIGHTGBM — CLEAN RESEARCH-GRADE VERSION
# Thermoelectric ZT Prediction
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    KFold
)

from sklearn.metrics import (
    mean_squared_error,
    r2_score
)

from lightgbm import LGBMRegressor

# ============================================================
# LOAD DATASET
# ============================================================

DATA_PATH = "/home/sudarshan/Documents/INFORMATICS/dataset/dataset/dataset_of_thermoelectric_figures.csv"

df = pd.read_csv(DATA_PATH)

print(f"\nLoaded Dataset : {df.shape}")

# ============================================================
# TARGET
# ============================================================

TARGET_COL = "ZT"

# ============================================================
# FEATURES TO DROP
# ============================================================

DROP_COLS = [

    # ========================================================
    # NON-NUMERIC / FORMULA
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

# ============================================================
# CREATE CLEAN FEATURE MATRIX
# ============================================================

X = df.drop(columns=[TARGET_COL] + DROP_COLS)

# ============================================================
# KEEP ONLY NUMERIC FEATURES
# ============================================================

X = X.select_dtypes(include=np.number)

# ============================================================
# HANDLE MISSING VALUES
# ============================================================

X = X.fillna(X.median(numeric_only=True))

# ============================================================
# TARGET VECTOR
# ============================================================

y = df[TARGET_COL]

# ============================================================
# FINAL DATASET INFO
# ============================================================

print("\nFinal Feature Shape:")

print(X.shape)

print("\nRemaining Missing Values:")

print(X.isnull().sum().sum())

# ============================================================
# FEATURE NAMES
# ============================================================

feature_names = X.columns.tolist()

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42
)

print(f"\nTrain Samples : {len(y_train)}")

print(f"Test Samples  : {len(y_test)}")

# ============================================================
# LIGHTGBM MODEL
# ============================================================

model = LGBMRegressor(

    n_estimators=300,

    learning_rate=0.05,

    max_depth=6,

    num_leaves=63,

    subsample=0.8,

    colsample_bytree=0.8,

    reg_alpha=0.1,

    reg_lambda=1.0,

    min_child_samples=20,

    random_state=42,

    n_jobs=-1,

    verbose=-1
)

# ============================================================
# CROSS VALIDATION
# ============================================================

print("\nRunning Cross Validation ...")

kf = KFold(

    n_splits=5,

    shuffle=True,

    random_state=42
)

cv_r2 = cross_val_score(

    model,

    X_train,

    y_train,

    cv=kf,

    scoring='r2'
)

# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining LightGBM Model ...")

model.fit(X_train, y_train)

# ============================================================
# PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)

# ============================================================
# METRICS
# ============================================================

r2 = r2_score(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

# ============================================================
# RESULTS
# ============================================================

print("\n" + "="*50)

print("LIGHTGBM RESULTS")

print("="*50)

print(f"CV R2 Mean : {cv_r2.mean():.4f}")

print(f"CV R2 Std  : {cv_r2.std():.4f}")

print(f"Test R2    : {r2:.4f}")

print(f"MSE        : {mse:.6f}")

print(f"RMSE       : {rmse:.6f}")

print("="*50)

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.Series(

    model.feature_importances_,

    index=feature_names

).sort_values(ascending=False)

print("\nTop 15 Features:\n")

print(importance.head(15))

# ============================================================
# PLOTS
# ============================================================

fig, axes = plt.subplots(

    1,

    3,

    figsize=(20, 6)
)

# ============================================================
# ACTUAL VS PREDICTED
# ============================================================

axes[0].scatter(

    y_test,

    y_pred,

    alpha=0.4,

    s=20
)

lims = [

    min(y_test.min(), y_pred.min()),

    max(y_test.max(), y_pred.max())
]

axes[0].plot(lims, lims, 'r--')

axes[0].set_xlabel("Actual ZT")

axes[0].set_ylabel("Predicted ZT")

axes[0].set_title(f"Actual vs Predicted\nR2 = {r2:.4f}")

# ============================================================
# METRICS BAR CHART
# ============================================================

metrics = ['R2', 'MSE', 'RMSE']

values = [r2, mse, rmse]

axes[1].bar(metrics, values)

axes[1].set_title("Performance Metrics")

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

top15 = importance.head(15)

axes[2].barh(

    top15.index[::-1],

    top15.values[::-1]
)

axes[2].set_title("Top 15 Feature Importances")

# ============================================================
# SAVE FIGURE
# ============================================================

plt.tight_layout()

plt.savefig(

    "/home/sudarshan/Documents/INFORMATICS/ML/lightgbm_clean_results.png",

    dpi=300,

    bbox_inches='tight'
)

print("\nFigure Saved Successfully")

plt.show()