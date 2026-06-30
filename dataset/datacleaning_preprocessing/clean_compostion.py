import pandas as pd
import re

# Load dataset
df = pd.read_csv("/home/sudarshan/Documents/inverse_design/data/processed/final_starrydata.csv")

print("Original shape:", df.shape)

# Function to clean composition
def clean_formula(comp):
    if pd.isna(comp):
        return None

    comp = str(comp)

    # Remove everything after 'with'
    comp = comp.split("with")[0]

    # Remove unwanted symbols
    comp = comp.replace(":", "")
    comp = comp.replace(" ", "")

    # Keep only letters and numbers
    comp = re.sub(r"[^A-Za-z0-9\.]", "", comp)

    return comp

# Apply cleaning
df["clean_formula"] = df["composition"].apply(clean_formula)

# Remove empty formulas
df = df.dropna(subset=["clean_formula"])

# Remove very small/invalid formulas
df = df[df["clean_formula"].str.len() > 2]

print("After cleaning:", df.shape)

# Save cleaned dataset
df.to_csv("/home/sudarshan/Documents/inverse_design/data/processed/cleaned_composition.csv", index=False)

print("✅ Composition cleaned and saved!")
