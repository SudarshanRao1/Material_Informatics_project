import pandas as pd

df1 = pd.read_csv("/home/sudarshan/Documents/inverse_design/data/processed/clean_starrydata.csv")
df2 = pd.read_csv("/home/sudarshan/Documents/inverse_design/data/processed/clean_starrydata2.csv")

combined = pd.concat([df1, df2], ignore_index=True)

# REMOVE DUPLICATES FIRST
combined = combined.drop_duplicates()

print("Final shape:", combined.shape)

# THEN SAVE
combined.to_csv("/home/sudarshan/Documents/inverse_design/data/processed/final_starrydata.csv", index=False)