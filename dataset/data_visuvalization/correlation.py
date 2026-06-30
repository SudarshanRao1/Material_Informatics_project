# ============================================================
# FULL RESEARCH-GRADE CORRELATION HEATMAP
# Large Figure + Full Matrix + Publication Quality
# ============================================================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import os

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(
    "/home/sudarshan/Documents/INFORMATICS/dataset/dataset/dataset_of_thermoelectric_figures.csv"
)

# ============================================================
# OUTPUT DIRECTORY
# ============================================================

save_folder = "figures"

os.makedirs(save_folder, exist_ok=True)

# ============================================================
# SELECT ONLY NUMERIC COLUMNS
# ============================================================

numeric_df = df.select_dtypes(include=np.number)

# ============================================================
# COMPUTE CORRELATION MATRIX
# ============================================================

corr = numeric_df.corr(method='pearson')

# ============================================================
# PLOT SETTINGS
# ============================================================

plt.figure(figsize=(40, 34))

# ============================================================
# HEATMAP
# ============================================================

heatmap = sns.heatmap(

    corr,

    cmap='coolwarm',

    vmin=-1,
    vmax=1,

    center=0,

    annot=False,

    linewidths=0.05,

    square=False,

    cbar_kws={
        "shrink": 0.8,
        "aspect": 30
    }

)

# ============================================================
# TITLE
# ============================================================

plt.title(
    "Correlation Heatmap",

    fontsize=40,

    fontweight='bold',

    pad=30
)

# ============================================================
# X-AXIS LABELS
# ============================================================

plt.xticks(

    rotation=45,

    ha='right',

    fontsize=11,

    fontweight='bold'
)

# ============================================================
# Y-AXIS LABELS
# ============================================================

plt.yticks(

    rotation=0,

    fontsize=11,

    fontweight='bold'
)

# ============================================================
# COLORBAR SETTINGS
# ============================================================

colorbar = heatmap.collections[0].colorbar

colorbar.ax.tick_params(labelsize=14)

# ============================================================
# LAYOUT
# ============================================================

plt.tight_layout()

# ============================================================
# SAVE FIGURE
# ============================================================

plt.savefig(

    os.path.join(
        save_folder,
        "full_research_correlation_heatmap.png"
    ),

    dpi=600,

    bbox_inches='tight'
)

# ============================================================
# SHOW FIGURE
# ============================================================

plt.show()

# ============================================================
# END
# ============================================================