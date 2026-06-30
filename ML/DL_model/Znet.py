"""
============================================================
  Thermoelectric Merit-ZT Prediction
  Model  : DEEP LEARNING — Neural Network (PyTorch)
  Dataset: dataset_of_thermoelectric_figures.csv
  Target : ZT  |  Clean features only (all leaky/redundant dropped)
  Runtime: ~3-5 minutes on Google Colab GPU
============================================================
"""

# ── Install (Colab already has these, just in case) ─────────────────
# !pip install torch scikit-learn pandas numpy matplotlib -q

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time, warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ── GPU check ────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {device}")
if device.type == "cuda":
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

# ════════════════════════════════════════════════════════════════════
#  STEP 1 — LOAD & CLEAN
# ════════════════════════════════════════════════════════════════════

# ── Update this path if running locally ─────────────────────────────
DATA_PATH = "dataset_of_thermoelectric_figures.csv"
TARGET_COL = "ZT"

DROP_COLS = [
    'composition',
    # ── Direct target leakage ────────────────────────────────────────
    'Seebeck coefficient',
    'Thermal conductivity',
    'Electrical conductivity',
    'reduced_ZT_T',
    'ZT_numerator_proxy',
    'power_factor_proxy',
    'sigma_over_kappa',
    'kappa_electronic_WF',
    'kappa_lattice_proxy',
    # ── Duplicate features ───────────────────────────────────────────
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
    # ── Low information ──────────────────────────────────────────────
    'Seebeck_sign',
    # ── Redundant log features ───────────────────────────────────────
    'log_electrical_conductivity',
    'log_thermal_conductivity',
    # ── Temperature transforms ───────────────────────────────────────
    'T_squared',
    'log_T',
    'T_times_avg_Z',
    'T_times_VEC',
    'T_over_Debye',
    'T_over_avg_melting_point',
    # ── Redundant abs features ───────────────────────────────────────
    'abs_Seebeck',
    'Seebeck_squared',
]

df = pd.read_csv(DATA_PATH)
print(f"Loaded   : {df.shape[0]:,} rows x {df.shape[1]} columns")

# Drop only columns that actually exist
drop_existing = [c for c in DROP_COLS if c in df.columns]
X = df.drop(columns=drop_existing + [TARGET_COL])
y = df[TARGET_COL].values.astype(np.float32)

X = X.fillna(X.median(numeric_only=True))
X = X.values.astype(np.float32)

print(f"Features : {X.shape[1]}  (after dropping {len(drop_existing)} redundant cols)")
print(f"ZT range : [{y.min():.4f} , {y.max():.4f}]  |  mean = {y.mean():.4f}")

# ════════════════════════════════════════════════════════════════════
#  STEP 2 — SPLIT & SCALE
# ════════════════════════════════════════════════════════════════════

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42
)

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train).astype(np.float32)
X_val_sc   = scaler.transform(X_val).astype(np.float32)
X_test_sc  = scaler.transform(X_test).astype(np.float32)

print(f"\nTrain : {len(y_train):,}  |  Val : {len(y_val):,}  |  Test : {len(y_test):,}")

# ── PyTorch tensors ──────────────────────────────────────────────────
def to_tensor(X, y):
    return TensorDataset(
        torch.tensor(X, dtype=torch.float32).to(device),
        torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)
    )

train_ds = to_tensor(X_train_sc, y_train)
val_ds   = to_tensor(X_val_sc,   y_val)
test_ds  = to_tensor(X_test_sc,  y_test)

train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=512)
test_loader  = DataLoader(test_ds,  batch_size=512)

# ════════════════════════════════════════════════════════════════════
#  STEP 3 — MODEL ARCHITECTURE
#  Deep Residual MLP — best suited for tabular regression
#  Input -> [Dense + BN + ReLU + Dropout] x blocks + Skip connections
# ════════════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))   # skip connection


class ZTNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, n_blocks=4, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.res_blocks(x)
        return self.head(x)


input_dim  = X_train_sc.shape[1]
model_nn   = ZTNet(input_dim=input_dim, hidden_dim=256, n_blocks=4, dropout=0.2).to(device)

total_params = sum(p.numel() for p in model_nn.parameters() if p.requires_grad)
print(f"\nModel    : ZTNet  |  Parameters : {total_params:,}")
print(f"Input dim: {input_dim}  ->  256 -> ResBlock x4 -> 64 -> 1")

# ════════════════════════════════════════════════════════════════════
#  STEP 4 — TRAINING CONFIG
# ════════════════════════════════════════════════════════════════════

EPOCHS    = 100
LR        = 1e-3
optimizer = torch.optim.AdamW(model_nn.parameters(), lr=LR, weight_decay=1e-4)
criterion = nn.MSELoss()

# Cosine annealing LR scheduler — smooth decay, no need to hand-tune
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ── Training loop ────────────────────────────────────────────────────
train_losses, val_losses = [], []
best_val_loss = float("inf")
best_state    = None

print(f"\nTraining for {EPOCHS} epochs  (batch=512, lr={LR}) ...\n")
t0 = time.time()

for epoch in range(1, EPOCHS + 1):
    # — Train —
    model_nn.train()
    batch_losses = []
    for Xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model_nn(Xb)
        loss = criterion(pred, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model_nn.parameters(), 1.0)
        optimizer.step()
        batch_losses.append(loss.item())
    train_loss = np.mean(batch_losses)

    # — Validate —
    model_nn.eval()
    val_batch = []
    with torch.no_grad():
        for Xb, yb in val_loader:
            val_batch.append(criterion(model_nn(Xb), yb).item())
    val_loss = np.mean(val_batch)

    scheduler.step()
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state    = {k: v.cpu().clone() for k, v in model_nn.state_dict().items()}

    if epoch % 10 == 0 or epoch == 1:
        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]
        print(f"  Epoch {epoch:>3}/{EPOCHS}  |  "
              f"Train MSE: {train_loss:.5f}  |  "
              f"Val MSE: {val_loss:.5f}  |  "
              f"LR: {lr_now:.2e}  |  {elapsed:.1f}s")

total_time = time.time() - t0
print(f"\nTraining done in {total_time:.1f}s  ({total_time/60:.1f} min)")

# ════════════════════════════════════════════════════════════════════
#  STEP 5 — EVALUATE ON TEST SET
# ════════════════════════════════════════════════════════════════════

model_nn.load_state_dict(best_state)   # restore best checkpoint
model_nn.to(device)
model_nn.eval()

all_preds, all_true = [], []
with torch.no_grad():
    for Xb, yb in test_loader:
        all_preds.append(model_nn(Xb).cpu().numpy())
        all_true.append(yb.cpu().numpy())

y_pred = np.concatenate(all_preds).flatten()
y_true = np.concatenate(all_true).flatten()

r2   = r2_score(y_true, y_pred)
mse  = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)

print("\n" + "="*50)
print("  DEEP LEARNING (ZTNet) — FINAL RESULTS")
print("="*50)
print(f"  Test R2   : {r2:.4f}")
print(f"  Test MSE  : {mse:.6f}")
print(f"  Test RMSE : {rmse:.6f}")
print("="*50)

# ════════════════════════════════════════════════════════════════════
#  STEP 6 — PLOTS
# ════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(19, 5))
fig.suptitle("Deep Learning (ZTNet — Residual MLP) — ZT Prediction",
             fontsize=14, fontweight="bold")

# — Actual vs Predicted —
axes[0].scatter(y_true, y_pred, alpha=0.3, color="#264653",
                edgecolors="none", s=15)
lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
axes[0].plot(lims, lims, "r--", lw=1.5, label="Ideal fit")
axes[0].set_xlabel("Actual ZT", fontsize=11)
axes[0].set_ylabel("Predicted ZT", fontsize=11)
axes[0].set_title(f"Actual vs Predicted\nR2 = {r2:.4f}", fontsize=11)
axes[0].legend(fontsize=9)

# — Loss curve —
axes[1].plot(train_losses, label="Train MSE", color="#2A9D8F", lw=1.5)
axes[1].plot(val_losses,   label="Val MSE",   color="#E76F51", lw=1.5)
axes[1].set_xlabel("Epoch", fontsize=11)
axes[1].set_ylabel("MSE Loss", fontsize=11)
axes[1].set_title("Training & Validation Loss", fontsize=11)
axes[1].legend(fontsize=9)

# — Metric bar chart : R2, MSE, RMSE —
metric_names = ["R2", "MSE", "RMSE"]
metric_vals  = [r2, mse, rmse]
bar_colors   = ["#264653", "#E9C46A", "#E76F51"]
bars = axes[2].bar(metric_names, metric_vals, color=bar_colors, alpha=0.85, width=0.4)
for bar, val in zip(bars, metric_vals):
    axes[2].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + max(metric_vals) * 0.01,
        f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold"
    )
axes[2].set_ylabel("Score / Error", fontsize=11)
axes[2].set_title("Performance Metrics\n(R2, MSE, RMSE)", fontsize=11)

plt.tight_layout()
plt.savefig("deeplearning_ZTNet_results.png", dpi=150, bbox_inches="tight")
print("\nPlot saved -> deeplearning_ZTNet_results.png")
plt.show()

