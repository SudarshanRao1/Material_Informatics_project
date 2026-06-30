'''show minimum  and maximum  of important columns.'''

import pandas as pd
df = pd.read_csv("/home/sudarshan/Downloads/asalu_emijarigindhi_dataset.csv")

print("Seebeck coefficient:")
print(df["Seebeck coefficient"].min(), df["Seebeck coefficient"].max())

print("\nElectrical conductivity:")
print(df["Electrical conductivity"].min(), df["Electrical conductivity"].max())

print("\nThermal conductivity:")
print(df["Thermal conductivity"].min(), df["Thermal conductivity"].max())

print("\nZT:")
print(df["ZT"].min(), df["ZT"].max())
print(df["Thermal conductivity"].sort_values(ascending=False).head(20))
