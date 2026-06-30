import pandas as pd

# Step 1: Load your dataset
df = pd.read_csv("/home/sudarshan/Documents/inverse_design/data/processed/final_ready_dataset.csv")   # change filename if needed

# Step 2: Check columns
print("Columns in dataset:\n", df.columns)

# Step 3: Drop useless columns (all-zero columns)
cols_to_drop = ['Seebeck coefficient', 'Electrical conductivity', 'Thermal conductivity']

# Only drop if they exist (safe)
df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

# Step 4: Verify removal
print("\nRemaining columns:\n", df.columns)

# Step 5: Check if any column still has all zeros
zero_cols = [col for col in df.columns if (df[col] == 0).all()]
print("\nColumns with all zero values:", zero_cols)

# Step 6: Final dataset shape
print("\nFinal dataset shape:", df.shape)

# Step 7: Save cleaned dataset
df.to_csv("/home/sudarshan/Documents/inverse_design/data/processed/the_final.csv", index=False)

print("\n✅ Dataset cleaned and saved as 'the_final.csv'")

