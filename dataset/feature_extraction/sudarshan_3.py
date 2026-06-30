import numpy as np
from pymatgen.core import Composition

def safe_get(el, prop):
    try:
        val = getattr(el, prop)
        return val if val is not None else None
    except:
        return None

def sudarshan_electronegitivity(comp):
    comp = Composition(comp)
    elements = comp.elements
    fractions = [comp.get_atomic_fraction(el) for el in elements]

    def weighted_avg(prop):
        vals = []
        weights = []
        for el, f in zip(elements, fractions):
            val = safe_get(el, prop)
            if val is not None:
                vals.append(val * f)
                weights.append(f)
        return sum(vals) / sum(weights) if weights else 0

    def values(prop):
        vals = []
        for el in elements:
            val = safe_get(el, prop)
            if val is not None:
                vals.append(val)
        return vals if vals else [0.0]

    features = {}
  #Electronegitivity
    vals = values("X")
    features["avg_X"] = weighted_avg("X")
    features["max_X"] = max(vals)
    features["min_X"] = min(vals)
    features["range_X"] = max(vals) - min(vals)
    features["std_X"] = np.std(vals)

    return features
