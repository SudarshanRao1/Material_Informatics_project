from mp_api.client import MPRester
import pandas as pd

API_KEY = "owzL4SdCjyp7ryLvug05R9ZN5ZFFNWrY"

with MPRester(API_KEY) as mpr:
    docs = mpr.summary.search(
        fields=[
            "material_id",
            "formula_pretty",
            "band_gap",
            "density",
            "energy_above_hull",
            "volume",
            "elements",
            "nsites"
        ],
        num_chunks=10
    )

data = []
for d in docs:
    data.append({
        "material_id": d.material_id,
        "formula": d.formula_pretty,
        "band_gap": d.band_gap,
        "density": d.density,
        "energy_above_hull": d.energy_above_hull,
        "volume": d.volume,
        "num_elements": len(d.elements),
        "nsites": d.nsites
    })

df = pd.DataFrame(data)
df.to_csv("../data/raw/materials_project.csv", index=False)

print("✅ Materials Project data saved!")

