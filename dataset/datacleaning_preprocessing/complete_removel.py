import pandas as pd
import numpy as np
from pymatgen.core import Composition

# Load dataset
df = pd.read_csv("/home/sudarshan/Documents/inverse_design/data/processed/cleaned_composition.csv")

print("Original dataset shape:", df.shape)


# -------------------------------
# STEP 1: BASIC CLEANING
# -------------------------------
def clean_formula(comp):
    comp = str(comp).strip()
    comp = comp.replace(" ", "")
    return comp

df["composition"] = df["composition"].apply(clean_formula)


# -------------------------------
# STEP 2: VALIDATION
# -------------------------------
def is_valid(comp):
    try:
        Composition(comp)
        return True
    except:
        return False

df["valid"] = df["composition"].apply(is_valid)

# Remove invalid
df = df[df["valid"] == True].copy()
df.drop(columns=["valid"], inplace=True)

print("After removing invalid:", df.shape)


# -------------------------------
# STEP 3: REMOVE SINGLE ELEMENT
# -------------------------------
def num_elements(comp):
    return len(Composition(comp).elements)

df["num_elements"] = df["composition"].apply(num_elements)

df = df[df["num_elements"] > 1].copy()
df.drop(columns=["num_elements"], inplace=True)

print("After removing single-element:", df.shape)


# -------------------------------
# STEP 4: REMOVE RARE COMPOSITIONS
# -------------------------------
# (Keep only compositions appearing at least 2 times)

counts = df["composition"].value_counts()

valid_comps = counts[counts >= 2].index

df = df[df["composition"].isin(valid_comps)].copy()

print("After removing rare compositions:", df.shape)


# -------------------------------
# STEP 5: REMOVE ZT OUTLIERS
# -------------------------------
# Assuming column name is "ZT"

Q1 = df["ZT"].quantile(0.25)
Q3 = df["ZT"].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[(df["ZT"] >= lower) & (df["ZT"] <= upper)].copy()

print("After removing ZT outliers:", df.shape)


# -------------------------------
# STEP 6: FINAL SAFETY CHECK
# -------------------------------
def final_check(comp):
    try:
        Composition(comp)
        return True
    except:
        return False

check = df["composition"].apply(final_check)

print("Any invalid left?", check.value_counts())


# -------------------------------
# STEP 7: SAVE FINAL DATASET
# -------------------------------
df.to_csv("/home/sudarshan/Documents/inverse_design/data/processed/ultra_clean_dataset.csv", index=False)

print("\n🔥 Dataset is now ULTRA CLEAN and ready for ML + Inverse Design 🔥")
