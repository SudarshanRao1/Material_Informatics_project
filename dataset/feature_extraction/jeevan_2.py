import numpy as np
from pymatgen.core import Composition

def safe_get(el, prop):
    try:
        val = getattr(el, prop)
        return val if val is not None else None
    except:
        return None

def jeevan_features_valence(comp):
    comp = Composition(comp)
    elements = comp.elements

    features = {}
# features relatedd to the reactivity of the composition
    valence = []
    for el in elements:
        val = safe_get(el, "group")
        if val is not None:
            valence.append(val)

    if valence:
        features["avg_valence_electrons"] = np.mean(valence)
        features["max_valence_electrons"] = max(valence)
        features["min_valence_electrons"] = min(valence)
        features["valence_range"] = max(valence) - min(valence)
        features["valence_std"] = np.std(valence)
    else:
        features.update({
            "avg_valence_electrons": 0,
            "max_valence_electrons": 0,
            "min_valence_electrons": 0,
            "valence_range": 0,
            "valence_std": 0
        })

    return features
