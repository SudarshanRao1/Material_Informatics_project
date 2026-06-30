''' using this code Remove impossible or suspicious values Show top 20 highest electrical conductivity values and removes them Check duplicate compositions at same temperature  remove duplicate rows
final missing value check'''

import pandas as pd

df = pd.read_csv("/home/sudarshan/Documents/myturn/final_thermoelectric_dataset.csv")

print("Original shape:", df.shape)

df = df[
    (df["Temperature"] > 0) &
    (df["ZT"] >= 0) &
    (df["Thermal conductivity"] > 0) &
    (df["Electrical conductivity"] > 0)
]

print("Shape after removing impossible values:", df.shape)

print("\nTop 20 Electrical Conductivity Values:")
print(df["Electrical conductivity"].sort_values(ascending=False).head(20))

Q1 = df["Electrical conductivity"].quantile(0.25)
Q3 = df["Electrical conductivity"].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[
    (df["Electrical conductivity"] >= lower) &
    (df["Electrical conductivity"] <= upper)
]

print("Shape after removing electrical conductivity outliers:", df.shape)

duplicates = df.duplicated(subset=["composition", "Temperature"], keep=False)

print("\nDuplicate rows based on composition and Temperature:")
print(df[duplicates].head(20))

print("\nNumber of duplicate rows:", duplicates.sum())

df = df.drop_duplicates(subset=["composition", "Temperature"])

print("Shape after removing duplicates:", df.shape)

print("\nMissing values in each column:")
print(df.isnull().sum())

print("\nFinal dataset statistics:")
print(df.describe())

output = "/home/sudarshan/Downloads/asalu_emijarigindhi_dataset.csv"
df.to_csv(output, index=False)

print("\nFinal curated dataset saved to:")
print(output)
print("Final shape:", df.shape)
