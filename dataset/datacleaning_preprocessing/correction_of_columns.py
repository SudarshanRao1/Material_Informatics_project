'''with this code i am taking out required columns from the raw dataset and saving them in the another file'''

import pandas as pd

df = pd.read_csv("/home/sudarshan/Downloads/20190711_preprocessing_interpolated_data.csv")

required_columns = [
    "composition",
    "ZT",
    "Temperature",
    "Seebeck coefficient",
    "Electrical conductivity",
    "Thermal conductivity"
]

clean_df = df.dropna(subset=required_columns)

clean_df = clean_df.drop_duplicates()

print("Final shape:", clean_df.shape)

clean_df.to_csv("/home/sudarshan/Documents/myturn/final_thermoelectric_dataset.csv", index=False)

print(clean_df.isnull().sum())
print(clean_df.describe())
