'''with this code i am taking out required columns from the raw dataset and saving them in the another file'''
import pandas as pd

df = pd.read_csv("/home/sudarshan/Documents/myturn/rawdata.csv")

columns_to_keep = [
    "composition",
    "ZT",
    "Temperature",
    "Seebeck coefficient",
    "Electrical conductivity",
    "Thermal conductivity"
]

df = df[columns_to_keep]

df.to_csv("only_main_columns.csv", index=False)
print(df.columns)
print("New shape:", df.shape)
