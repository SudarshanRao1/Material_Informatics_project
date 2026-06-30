'''removing sample_ID form the dataset'''

import pandas as pd

df = pd.read_csv("/home/sudarshan/Documents/myturn/datasets/final_featured_dataset.csv")

# Remove sampleid column
df = df.drop(columns=["sampleid"])

df.to_csv("/home/sudarshan/Documents/myturn/final_featured_ID_dataset.csv", index=False)

print(df.head())
print(df.shape)
