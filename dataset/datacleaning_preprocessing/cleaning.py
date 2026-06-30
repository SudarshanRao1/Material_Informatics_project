import pandas as pd

# Load feature dataset
df = pd.read_csv("/home/sudarshan/Documents/inverse_design/data/processed/final_45_features_dataset.csv")

print("Before cleaning NaNs:", df.isnull().sum().sum())

# Fill NaNs
df = df.fillna(0)

print("After cleaning NaNs:", df.isnull().sum().sum())

# Save clean dataset
df.to_csv("/home/sudarshan/Documents/inverse_design/data/processed/final_45_features_dataset_clean.csv", index=False)

print("🔥 Clean dataset saved 🔥")
