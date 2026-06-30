import numpy as np
from pymatgen.core import Composition

def safe_get(el, prop):
    try:
        val = getattr(el, prop)
        return val if val is not None else None
    except:
        return None

def jeevan_features_ionization(comp):
    comp = Composition(comp)
    elements = comp.elements

    features = {}
# stability of the composition , that is why the ionnization of the elements were added
    ion = []
    for el in elements:
        val = safe_get(el, "ionization_energy")
        if val is not None:
            ion.append(val)

    if ion:
        features["avg_ionization_energy"] = np.mean(ion)
        features["max_ionization_energy"] = max(ion)
        features["min_ionization_energy"] = min(ion)
        features["ionization_range"] = max(ion) - min(ion)
        features["ionization_std"] = np.std(ion)
    else:
        features.update({
            "avg_ionization_energy": 0,
            "max_ionization_energy": 0,
            "min_ionization_energy": 0,
            "ionization_range": 0,
            "ionization_std": 0
        })

    return features
