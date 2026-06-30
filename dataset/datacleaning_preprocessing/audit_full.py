import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("/home/sudarshan/Documents/inverse_design/data/processed/final_ready_dataset.csv")

print("\n================ DATASET AUDIT ================\n")

# -------------------------------
# BASIC INFO
# -------------------------------
print("Shape:", df.shape)

# -------------------------------
# 1. NaN CHECK
# -------------------------------
nan_count = df.isnull().sum().sum()
print("\nTotal NaN values:", nan_count)

# -------------------------------
# 2. DUPLICATE CHECK
# -------------------------------
dup_count = df.duplicated().sum()
print("Duplicate rows:", dup_count)

# -------------------------------
# 3. INFINITE VALUES CHECK
# -------------------------------
inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
print("Infinite values:", inf_count)

# -------------------------------
# 4. CONSTANT FEATURES
# -------------------------------
constant_cols = [col for col in df.columns if df[col].nunique() == 1]
print("\nConstant columns:", constant_cols)
print("Number of constant columns:", len(constant_cols))

# -------------------------------
# 5. TARGET (ZT) CHECK
# -------------------------------
print("\nZT Statistics:")
print(df["ZT"].describe())

# -------------------------------
# 6. EXTREME VALUES CHECK
# -------------------------------
print("\nTop 5 MAX values:")
print(df.max(numeric_only=True).sort_values(ascending=False).head(5))

print("\nTop 5 MIN values:")
print(df.min(numeric_only=True).sort_values().head(5))

# -------------------------------
# 7. FEATURE DISTRIBUTION CHECK
# -------------------------------
print("\nFeature variance check (low variance features):")
low_var_cols = df.var(numeric_only=True)
low_var_cols = low_var_cols[low_var_cols < 1e-6]
print(low_var_cols)

# -------------------------------
# FINAL VERDICT
# -------------------------------
print("\n================ FINAL VERDICT ================\n")

if nan_count == 0 and inf_count == 0:
    print("✅ No NaN or infinite values")
else:
    print("❌ Fix NaN or infinite values")

if dup_count == 0:
    print("✅ No duplicate rows")
else:
    print("⚠️ Duplicates present (optional to remove)")

if len(constant_cols) == 0:
    print("✅ No useless constant features")
else:
    print("⚠️ Drop constant features:", constant_cols)

if df["ZT"].min() < 0:
    print("⚠️ Negative ZT values found (check dataset)")
else:
    print("✅ ZT values look valid")

print("\n🔥 DATASET AUDIT COMPLETE 🔥")
