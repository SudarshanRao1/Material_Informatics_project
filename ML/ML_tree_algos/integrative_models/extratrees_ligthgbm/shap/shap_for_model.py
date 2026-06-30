# ============================================================
# STACKED ENSEMBLE
# ExtraTrees + LightGBM
# Thermoelectric ZT Prediction
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
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

    mean_squared_error
)

from sklearn.ensemble import (

    ExtraTreesRegressor,

    StackingRegressor
)

from sklearn.linear_model import Ridge

# ============================================================
# LIGHTGBM
# ============================================================

from lightgbm import LGBMRegressor

# ============================================================
# LOAD DATASET
# ============================================================

DATA_PATH = "/home/sudarshan/Documents/INFORMATICS/dataset/dataset/dataset_of_thermoelectric_figures.csv"

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
# EXTRA TREES MODEL
# ============================================================

extra_trees = ExtraTreesRegressor(

    n_estimators=300,

    max_features='sqrt',

    random_state=42,

    n_jobs=-1
)

# ============================================================
# LIGHTGBM MODEL
# ============================================================

lightgbm = LGBMRegressor(

    n_estimators=300,

    learning_rate=0.05,

    max_depth=6,

    num_leaves=63,

    subsample=0.8,

    colsample_bytree=0.8,

    random_state=42,

    n_jobs=-1,

    verbose=-1
)

# ============================================================
# STACKING ENSEMBLE
# ============================================================

stack_model = StackingRegressor(

    estimators=[

        ('et', extra_trees),

        ('lgbm', lightgbm)
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

# ============================================================
# RESULTS
# ============================================================

print("\n" + "="*55)

print("EXTRATREES + LIGHTGBM ENSEMBLE RESULTS")

print("="*55)

print(f"CV R2 Mean : {cv_scores.mean():.4f}")

print(f"CV R2 Std  : {cv_scores.std():.4f}")

print(f"Test R2    : {r2:.4f}")

print(f"MSE        : {mse:.6f}")

print(f"RMSE       : {rmse:.6f}")

print("="*55)

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

    f"ExtraTrees + LightGBM Ensemble\nR2 = {r2:.4f}",

    fontsize=18,

    fontweight='bold'
)

plt.grid(alpha=0.3)

plt.tight_layout()

# ============================================================
# SAVE FIGURE
# ============================================================

plt.savefig(

    "/home/sudarshan/Downloads/extratrees_lightgbm_ensemble.png",

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

    "/home/sudarshan/Downloads/residual_plot.png",

    dpi=300,

    bbox_inches='tight'
)

plt.show()

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

# Extract trained models from stacking ensemble

trained_et = stack_model.named_estimators_['et']

trained_lgbm = stack_model.named_estimators_['lgbm']

# ============================================================
# GET IMPORTANCES
# ============================================================

et_importance = trained_et.feature_importances_

lgbm_importance = trained_lgbm.feature_importances_

# ============================================================
# NORMALIZE
# ============================================================

et_importance = et_importance / et_importance.sum()

lgbm_importance = lgbm_importance / lgbm_importance.sum()

# ============================================================
# COMBINED IMPORTANCE
# ============================================================

combined_importance = (

    et_importance +

    lgbm_importance

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

    "/home/sudarshan/Downloads/ensemble_feature_importance.png",

    dpi=300,

    bbox_inches='tight'
)

plt.show()
# ============================================================
# END OF CODE
# ============================================================


# ============================================================
# SHAP ANALYSIS
# ============================================================

print("\nRunning SHAP Analysis ...")

# ============================================================
# USE TRAINED LIGHTGBM MODEL
# ============================================================

trained_lgbm = stack_model.named_estimators_['lgbm']

# ============================================================
# CREATE SHAP EXPLAINER
# ============================================================

explainer = shap.TreeExplainer(trained_lgbm)

# ============================================================
# SAMPLE DATA
# (important for speed)
# ============================================================

X_shap = X_test.sample(

    n=1000,

    random_state=42
)

# ============================================================
# COMPUTE SHAP VALUES
# ============================================================

shap_values = explainer.shap_values(X_shap)

print("SHAP Values Computed Successfully")

# ============================================================
# 1. SHAP SUMMARY PLOT
# ============================================================

plt.figure()

shap.summary_plot(

    shap_values,

    X_shap,

    show=False
)

plt.title(

    "SHAP Summary Plot",

    fontsize=18,

    fontweight='bold'
)

plt.tight_layout()

# ============================================================
# SAVE
# ============================================================

plt.savefig(

    "/home/sudarshan/Downloads/shap_summary_plot.png",

    dpi=300,

    bbox_inches='tight'
)

plt.show()

# ============================================================
# 2. SHAP BAR PLOT
# ============================================================

plt.figure()

shap.summary_plot(

    shap_values,

    X_shap,

    plot_type="bar",

    show=False
)

plt.title(

    "SHAP Feature Importance",

    fontsize=18,

    fontweight='bold'
)

plt.tight_layout()

# ============================================================
# SAVE
# ============================================================

plt.savefig(

    "/home/sudarshan/Downloads/shap_bar_plot.png",

    dpi=300,

    bbox_inches='tight'
)

plt.show()

# ============================================================
# 3. SHAP DEPENDENCE PLOT
# ============================================================

# Most important feature

top_feature = importance_df.iloc[0]['Feature']

plt.figure()

shap.dependence_plot(

    top_feature,

    shap_values,

    X_shap,

    show=False
)

plt.title(

    f"SHAP Dependence Plot : {top_feature}",

    fontsize=16,

    fontweight='bold'
)

plt.tight_layout()

# ============================================================
# SAVE
# ============================================================

plt.savefig(

    "/home/sudarshan/Downloads/shap_dependence_plot.png",

    dpi=300,

    bbox_inches='tight'
)

plt.show()

print("\nSHAP Analysis Completed Successfully")


# ============================================================
# SHAP ANALYSIS — EXTRATREES
# ============================================================

print("\nRunning ExtraTrees SHAP Analysis ...")

# ============================================================
# USE TRAINED EXTRATREES MODEL
# ============================================================

trained_et = stack_model.named_estimators_['et']

# ============================================================
# CREATE SHAP EXPLAINER
# ============================================================

explainer_et = shap.TreeExplainer(trained_et)

# ============================================================
# SAMPLE TEST DATA
# ============================================================

X_shap_et = X_test.sample(

    n=1000,

    random_state=42
)

# ============================================================
# COMPUTE SHAP VALUES
# ============================================================

shap_values_et = explainer_et.shap_values(X_shap_et)

print("ExtraTrees SHAP Values Computed Successfully")

# ============================================================
# 1. SHAP SUMMARY PLOT
# ============================================================

plt.figure()

shap.summary_plot(

    shap_values_et,

    X_shap_et,

    show=False
)

plt.title(

    "ExtraTrees SHAP Summary Plot",

    fontsize=18,

    fontweight='bold'
)

plt.tight_layout()

# ============================================================
# SAVE
# ============================================================

plt.savefig(

    "/home/sudarshan/Downloads/extratrees_shap_summary.png",

    dpi=300,

    bbox_inches='tight'
)

plt.show()

# ============================================================
# 2. SHAP BAR PLOT
# ============================================================

plt.figure()

shap.summary_plot(

    shap_values_et,

    X_shap_et,

    plot_type="bar",

    show=False
)

plt.title(

    "ExtraTrees SHAP Feature Importance",

    fontsize=18,

    fontweight='bold'
)

plt.tight_layout()

# ============================================================
# SAVE
# ============================================================

plt.savefig(

    "/home/sudarshan/Downloads/extratrees_shap_bar.png",

    dpi=300,

    bbox_inches='tight'
)

plt.show()

# ============================================================
# 3. SHAP DEPENDENCE PLOT
# ============================================================

top_feature_et = importance_df.iloc[0]['Feature']

plt.figure()

shap.dependence_plot(

    top_feature_et,

    shap_values_et,

    X_shap_et,

    show=False
)

plt.title(

    f"ExtraTrees SHAP Dependence Plot : {top_feature_et}",

    fontsize=16,

    fontweight='bold'
)

plt.tight_layout()

# ============================================================
# SAVE
# ============================================================

plt.savefig(

    "/home/sudarshan/Downloads/extratrees_shap_dependence.png",

    dpi=300,

    bbox_inches='tight'
)

plt.show()

print("\nExtraTrees SHAP Analysis Completed Successfully")
