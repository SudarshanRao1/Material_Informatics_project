# ============================================================
# STACKED ENSEMBLE
# XGBoost + Gradient Boosting
# Thermoelectric ZT Prediction
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# ============================================================
# SCIKIT-LEARN
# ============================================================

from sklearn.model_selection import (

    train_test_split,

    cross_val_score,

    KFold
)

from sklearn.metrics import (

    r2_score,

    mean_squared_error,

    mean_absolute_error
)

from sklearn.ensemble import (

    GradientBoostingRegressor,

    StackingRegressor
)

from sklearn.linear_model import Ridge

# ============================================================
# XGBOOST
# ============================================================

from xgboost import XGBRegressor

# ============================================================
# LOAD DATASET
# ============================================================

DATA_PATH = "dataset_of_thermoelectric_figures.csv"

TARGET_COL = "ZT"

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print(f"\nLoaded Dataset : {df.shape}")

# ============================================================
# CLEAN RESEARCH-GRADE DROP LIST
# ============================================================

DROP_COLS = [

    # ========================================================
    # NON NUMERIC
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
# CREATE FEATURE MATRIX
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
# FEATURE NAMES
# ============================================================

feature_names = X.columns.tolist()

print(f"\nRemaining Features : {len(feature_names)}")

print(f"Final Dataset Shape : {X.shape}")

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
# XGBOOST MODEL
# ============================================================

xgboost = XGBRegressor(

    n_estimators=300,

    learning_rate=0.05,

    max_depth=6,

    subsample=0.8,

    colsample_bytree=0.8,

    random_state=42,

    n_jobs=-1
)

# ============================================================
# GRADIENT BOOSTING MODEL
# ============================================================

gbm = GradientBoostingRegressor(

    n_estimators=300,

    learning_rate=0.05,

    max_depth=5,

    random_state=42
)

# ============================================================
# STACKING ENSEMBLE
# ============================================================

stack_model = StackingRegressor(

    estimators=[

        ('xgb', xgboost),

        ('gbm', gbm)
    ],

    # Meta learner

    final_estimator=Ridge(alpha=1.0),

    cv=5,

    n_jobs=-1,

    passthrough=False
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

cv_scores = cross_val_score(

    stack_model,

    X_train,

    y_train,

    cv=kf,

    scoring='r2',

    n_jobs=-1
)

# ============================================================
# TRAIN STACK MODEL
# ============================================================

print("\nTraining Ensemble Model ...")

stack_model.fit(X_train, y_train)

# ============================================================
# PREDICTIONS
# ============================================================

y_pred = stack_model.predict(X_test)

# ============================================================
# METRICS
# ============================================================

r2 = r2_score(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

mae = mean_absolute_error(y_test, y_pred)

# ============================================================
# RESULTS
# ============================================================

print("\n" + "="*55)

print("XGBOOST + GBM ENSEMBLE RESULTS")

print("="*55)

print(f"CV R2 Mean : {cv_scores.mean():.4f}")

print(f"CV R2 Std  : {cv_scores.std():.4f}")

print(f"Test R2    : {r2:.4f}")

print(f"MAE        : {mae:.6f}")

print(f"MSE        : {mse:.6f}")

print(f"RMSE       : {rmse:.6f}")

print("="*55)

# ============================================================
# METRICS SUMMARY BAR CHART
# ============================================================

metrics_names  = ['R2', 'MAE', 'RMSE', 'CV Mean', 'CV Std']

metrics_values = [r2, mae, rmse, cv_scores.mean(), cv_scores.std()]

colors = ['#2ecc71', '#e74c3c', '#e67e22', '#3498db', '#9b59b6']

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(metrics_names, metrics_values, color=colors, edgecolor='black', linewidth=0.8)

for bar, val in zip(bars, metrics_values):

    ax.text(

        bar.get_x() + bar.get_width() / 2,

        bar.get_height() + 0.005,

        f"{val:.4f}",

        ha='center',

        va='bottom',

        fontsize=12,

        fontweight='bold'
    )

ax.set_ylabel("Value", fontsize=14, fontweight='bold')

ax.set_title(

    f"XGBoost + GBM Ensemble -- Metrics Summary\nCV R2 = {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}",

    fontsize=16,

    fontweight='bold'
)

ax.set_ylim(0, max(metrics_values) * 1.2)

ax.grid(axis='y', alpha=0.3)

plt.tight_layout()

plt.savefig(

    "metrics_summary.png",

    dpi=300,

    bbox_inches='tight'
)

print("\nMetrics Summary Figure Saved")

plt.show()

# ============================================================
# CV FOLD SCORES PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5))

fold_nums = [f"Fold {i+1}" for i in range(len(cv_scores))]

bar_colors = ['#3498db' if s >= cv_scores.mean() else '#e74c3c' for s in cv_scores]

ax.bar(fold_nums, cv_scores, color=bar_colors, edgecolor='black', linewidth=0.8)

ax.axhline(

    y=cv_scores.mean(),

    color='green',

    linestyle='--',

    linewidth=2,

    label=f"Mean R2 = {cv_scores.mean():.4f}"
)

ax.fill_between(

    range(len(cv_scores)),

    cv_scores.mean() - cv_scores.std(),

    cv_scores.mean() + cv_scores.std(),

    alpha=0.15,

    color='green',

    label=f"+/- 1 STD ({cv_scores.std():.4f})"
)

for i, (x, s) in enumerate(zip(fold_nums, cv_scores)):

    ax.text(i, s + 0.002, f"{s:.4f}", ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xlabel("CV Fold", fontsize=13, fontweight='bold')

ax.set_ylabel("R2 Score", fontsize=13, fontweight='bold')

ax.set_title("5-Fold Cross Validation R2 Scores", fontsize=16, fontweight='bold')

ax.set_ylim(0, 1.05)

ax.legend(fontsize=11)

ax.grid(axis='y', alpha=0.3)

plt.tight_layout()

plt.savefig(

    "cv_fold_scores.png",

    dpi=300,

    bbox_inches='tight'
)

print("CV Fold Scores Figure Saved")

plt.show()

# ============================================================
# ACTUAL VS PREDICTED
# ============================================================

plt.figure(figsize=(10, 8))

plt.scatter(

    y_test,

    y_pred,

    alpha=0.4,

    s=20
)

lims = [

    min(y_test.min(), y_pred.min()),

    max(y_test.max(), y_pred.max())
]

plt.plot(

    lims,

    lims,

    'r--',

    linewidth=2
)

# Annotate metrics on plot

textstr = f"R² = {r2:.4f}\nMAE = {mae:.4f}\nRMSE = {rmse:.4f}\nCV R² = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}"

props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')

plt.gca().text(

    0.05, 0.95, textstr,

    transform=plt.gca().transAxes,

    fontsize=11,

    verticalalignment='top',

    bbox=props
)

plt.xlabel(

    "Actual ZT",

    fontsize=14,

    fontweight='bold'
)

plt.ylabel(

    "Predicted ZT",

    fontsize=14,

    fontweight='bold'
)

plt.title(

    f"XGBoost + GBM Ensemble\nR2 = {r2:.4f}",

    fontsize=18,

    fontweight='bold'
)

plt.grid(alpha=0.3)

plt.tight_layout()

# ============================================================
# SAVE FIGURE
# ============================================================

plt.savefig(

    "xgb_gbm_ensemble.png",

    dpi=300,

    bbox_inches='tight'
)

print("\nFigure Saved Successfully")

# ============================================================
# SHOW PLOT
# ============================================================

plt.show()

# ============================================================
# RESIDUAL PLOT
# ============================================================

residuals = y_test - y_pred

plt.figure(figsize=(10, 8))

plt.scatter(

    y_pred,

    residuals,

    alpha=0.4,

    s=20
)

plt.axhline(

    y=0,

    color='red',

    linestyle='--',

    linewidth=2
)

plt.xlabel(

    "Predicted ZT",

    fontsize=14,

    fontweight='bold'
)

plt.ylabel(

    "Residuals",

    fontsize=14,

    fontweight='bold'
)

plt.title(

    "Residual Plot",

    fontsize=18,

    fontweight='bold'
)

plt.grid(alpha=0.3)

plt.tight_layout()

# ============================================================
# SAVE
# ============================================================

plt.savefig(

    "residual_plot.png",

    dpi=300,

    bbox_inches='tight'
)

plt.show()

## ============================================================
# FEATURE IMPORTANCE
# ============================================================

# Extract trained models from stacking ensemble

trained_xgb = stack_model.named_estimators_['xgb']

trained_gbm = stack_model.named_estimators_['gbm']

# ============================================================
# GET IMPORTANCES
# ============================================================

xgb_importance = trained_xgb.feature_importances_

gbm_importance = trained_gbm.feature_importances_

# ============================================================
# NORMALIZE
# ============================================================

xgb_importance = xgb_importance / xgb_importance.sum()

gbm_importance = gbm_importance / gbm_importance.sum()

# ============================================================
# COMBINED IMPORTANCE
# ============================================================

combined_importance = (

    xgb_importance +

    gbm_importance

) / 2

# ============================================================
# DATAFRAME
# ============================================================

importance_df = pd.DataFrame({

    'Feature': feature_names,

    'Importance': combined_importance
})

importance_df = importance_df.sort_values(

    by='Importance',

    ascending=False
)

top15 = importance_df.head(15)

# ============================================================
# PLOT
# ============================================================

plt.figure(figsize=(12, 10))

plt.barh(

    top15['Feature'][::-1],

    top15['Importance'][::-1]
)

plt.xlabel(

    "Average Importance",

    fontsize=14,

    fontweight='bold'
)

plt.ylabel(

    "Features",

    fontsize=14,

    fontweight='bold'
)

plt.title(

    "Top 15 Ensemble Feature Importances",

    fontsize=18,

    fontweight='bold'
)

plt.tight_layout()

# ============================================================
# SAVE
# ============================================================

plt.savefig(

    "ensemble_feature_importance.png",

    dpi=300,

    bbox_inches='tight'
)

plt.show()
# ============================================================
# END OF CODE
# ============================================================