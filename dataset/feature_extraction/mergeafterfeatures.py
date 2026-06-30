import pandas as pd

from sudarshan_1 import sudarshan_atomic_numbers
from sudarshan_2 import sudarshan_atomic_mass
from sudarshan_3 import sudarshan_electronegitivity

from jeevan_1 import jeevan_features_block
from jeevan_2 import jeevan_features_valence
from jeevan_3 import jeevan_features_ionization

from shatrujit_1 import shatrujit_features_composition
from shatrujit_2 import shatrujit_features_differences
from shatrujit_3 import shatrujit_features_ratios

df = pd.read_csv("/home/sudarshan/Documents/myturn/datasets/final_curated_thermoelectric_dataset_cleaned.csv")

# sudarshan's features
p1_Z    = pd.DataFrame(df["composition"].apply(sudarshan_atomic_numbers).tolist())
p1_mass = pd.DataFrame(df["composition"].apply(sudarshan_atomic_mass).tolist())
p1_X    = pd.DataFrame(df["composition"].apply(sudarshan_electronegitivity).tolist())

# jeevan's features
p2_block      = pd.DataFrame(df["composition"].apply(jeevan_features_block).tolist())
p2_valence    = pd.DataFrame(df["composition"].apply(jeevan_features_valence).tolist())
p2_ionization = pd.DataFrame(df["composition"].apply(jeevan_features_ionization).tolist())

# shatrujit's features
p3_composition  = pd.DataFrame(df["composition"].apply(shatrujit_features_composition).tolist())
p3_differences  = pd.DataFrame(df["composition"].apply(shatrujit_features_differences).tolist())
p3_ratios       = pd.DataFrame(df["composition"].apply(shatrujit_features_ratios).tolist())

final_df = pd.concat(
    [df, p1_Z, p1_mass, p1_X,
     p2_block, p2_valence, p2_ionization,
     p3_composition, p3_differences, p3_ratios],
    axis=1
)

print(final_df.shape)

final_df.to_csv("/home/sudarshan/Documents/myturn/datasets/final_featured_ID_dataset.csv", index=False)
