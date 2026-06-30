'''using this code we are removing rows that doesn't contain ZT and Composition.'''

import pandas as pd

df = pd.read_csv("/home/sudarshan/Documents/myturn/only_main_columns.csv")

df = df.dropna(subset=["composition", "ZT"])

print("Shape after keeping composition and ZT:", df.shape)

df.to_csv("/home/sudarshan/Documents/myturn/curated_step2.csv", index=False)
