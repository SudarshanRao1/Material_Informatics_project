# ============================================================
# PCA ANALYSIS FROM SCRATCH
# Thermoelectric Dataset Visualization
# Research-Grade Implementation
# ============================================================

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import os

# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv(
    "/home/sudarshan/Documents/INFORMATICS/dataset/dataset/dataset_of_thermoelectric_figures.csv"
)

# ============================================================
# 3. OUTPUT DIRECTORY
# ============================================================

save_folder = "/home/sudarshan/Documents/INFORMATICS/dataset/data_visuvalization"

os.makedirs(save_folder, exist_ok=True)

# ============================================================
# 4. SELECT NUMERIC COLUMNS ONLY
# ============================================================

numeric_df = df.select_dtypes(include=np.number)

# ============================================================
# 5. CHOOSE TARGET COLUMN
# ============================================================

# Replace if your target column is different

target_column = 'ZT'

# ============================================================
# 6. SEPARATE FEATURES AND TARGET
# ============================================================

X = numeric_df.drop(columns=[target_column])

y = numeric_df[target_column]

# ============================================================
# 7. HANDLE MISSING VALUES
# ============================================================

# Fill missing values using column mean

X = X.fillna(X.mean())

# ============================================================
# 8. STANDARDIZE FEATURES
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ============================================================
# 9. APPLY PCA
# ============================================================

pca = PCA(n_components=2)

pca_result = pca.fit_transform(X_scaled)

# ============================================================
# 10. CREATE PCA DATAFRAME
# ============================================================

pca_df = pd.DataFrame({

    'PC1': pca_result[:, 0],

    'PC2': pca_result[:, 1],

    target_column: y.values
})

# ============================================================
# 11. EXPLAINED VARIANCE
# ============================================================

explained_variance = pca.explained_variance_ratio_

print("\nExplained Variance Ratio:\n")

print(f"PC1 : {explained_variance[0]:.4f}")

print(f"PC2 : {explained_variance[1]:.4f}")

print(
    f"\nTotal Explained Variance : "
    f"{explained_variance.sum():.4f}"
)

# ============================================================
# 12. PCA SCATTER PLOT
# ============================================================

plt.figure(figsize=(14, 12))

scatter = plt.scatter(

    pca_df['PC1'],

    pca_df['PC2'],

    c=pca_df[target_column],

    cmap='viridis',

    s=90,

    alpha=0.8
)

# ============================================================
# 13. LABELS
# ============================================================

plt.xlabel(

    f"PC1 ({explained_variance[0] * 100:.2f}% Variance)",

    fontsize=18,

    fontweight='bold'
)

plt.ylabel(

    f"PC2 ({explained_variance[1] * 100:.2f}% Variance)",

    fontsize=18,

    fontweight='bold'
)

# ============================================================
# 14. TITLE
# ============================================================

plt.title(

    "PCA Projection of Thermoelectric Dataset",

    fontsize=26,

    fontweight='bold',

    pad=20
)

# ============================================================
# 15. COLORBAR
# ============================================================

cbar = plt.colorbar(scatter)

cbar.set_label(

    target_column,

    fontsize=16,

    fontweight='bold'
)

# ============================================================
# 16. GRID
# ============================================================

plt.grid(alpha=0.3)

# ============================================================
# 17. LAYOUT
# ============================================================

plt.tight_layout()

# ============================================================
# 18. SAVE FIGURE
# ============================================================

plt.savefig(

    os.path.join(
        save_folder,
        "pca_projection.png"
    ),

    dpi=600,

    bbox_inches='tight'
)

# ============================================================
# 19. SHOW PLOT
# ============================================================

plt.show()

# ============================================================
# 20. PCA EXPLAINED VARIANCE CURVE
# ============================================================

# PCA with all components

pca_full = PCA()

pca_full.fit(X_scaled)

variance = pca_full.explained_variance_ratio_

# ============================================================
# 21. CUMULATIVE VARIANCE
# ============================================================

cumulative_variance = np.cumsum(variance)

# ============================================================
# 22. PLOT EXPLAINED VARIANCE
# ============================================================

plt.figure(figsize=(14, 10))

plt.plot(

    range(1, len(cumulative_variance) + 1),

    cumulative_variance,

    marker='o',

    linewidth=3
)

# ============================================================
# 23. LABELS
# ============================================================

plt.xlabel(

    "Number of Principal Components",

    fontsize=18,

    fontweight='bold'
)

plt.ylabel(

    "Cumulative Explained Variance",

    fontsize=18,

    fontweight='bold'
)

# ============================================================
# 24. TITLE
# ============================================================

plt.title(

    "PCA Explained Variance Curve",

    fontsize=26,

    fontweight='bold',

    pad=20
)

# ============================================================
# 25. GRID
# ============================================================

plt.grid(alpha=0.3)

# ============================================================
# 26. LAYOUT
# ============================================================

plt.tight_layout()

# ============================================================
# 27. SAVE FIGURE
# ============================================================

plt.savefig(

    os.path.join(
        save_folder,
        "pca_explained_variance.png"
    ),

    dpi=600,

    bbox_inches='tight'
)

# ============================================================
# 28. SHOW PLOT
# ============================================================

plt.show()

# ============================================================
# END OF CODE
# ============================================================