'''setting the temprature limit'''

import pandas as pd

df = pd.read_csv("/home/sudarshan/Downloads/asalu_emijarigindhi_dataset.csv")

print("Original shape:", df.shape)

df = df[df["Thermal conductivity"] <= 100]

print("Shape after removing thermal conductivity outliers:", df.shape)

print("\nThermal conductivity range:")
print(df["Thermal conductivity"].min(), df["Thermal conductivity"].max())

print("\nTop 20 Thermal Conductivity Values:")
print(df["Thermal conductivity"].sort_values(ascending=False).head(20))

output_path = "/home/sudarshan/Documents/myturn/thermoelectric_dataset.csv"
df.to_csv(output_path, index=False)

print("\nUpdated dataset saved to:")
print(output_path)
print("Final shape:", df.shape)
