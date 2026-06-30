import pandas as pd

# Load dataset
df = pd.read_csv("/home/sudarshan/Documents/inverse_design/data/raw/20191119_rawdata.csv")

print("Original shape:", df.shape)

# Keep only required columns
cols_to_keep = [
    "composition",
    "ZT",
    "Temperature",
    "Seebeck coefficient",
    "Electrical conductivity",
    "Thermal conductivity"
]

# Keep only available columns
cols_to_keep = [col for col in cols_to_keep if col in df.columns]

df = df[cols_to_keep]

# Remove rows without ZT
df = df.dropna(subset=["ZT"])

# Remove rows without composition
df = df.dropna(subset=["composition"])

# Reset index
df = df.reset_index(drop=True)

print("Cleaned shape:", df.shape)
print(df.head())

# Save cleaned file
df.to_csv("/home/sudarshan/Documents/inverse_design/data/processed/clean_starrydata.csv", index=False)

print("✅ Clean dataset saved!")
