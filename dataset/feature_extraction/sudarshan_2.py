import numpy as np
from pymatgen.core import Composition
# this code will add the features related to atomic mass of the composition
def safe_get(el, prop):
    try:
        val = getattr(el, prop)
        return val if val is not None else None
    except:
        return None

def sudarshan_atomic_mass(comp):
    comp=Composition(comp)
    elements=comp.elements
    fractions=[comp.get_atomic_fraction(el) for el in elements]

    def weighted_avg(prop):
        vals=[]
        weights=[]
        for el, f in zip(elements, fractions):
            val=safe_get(el, prop)
            if val is not None:
                vals.append(val * f)
                weights.append(f)
        return sum(vals) / sum(weights) if weights else 0

    def values(prop):
        vals=[]
        for el in elements:
            val = safe_get(el, prop)
            if val is not None:
                vals.append(val)
        return vals if vals else [0.0]

    features = {}
    #atomic mass
    vals = values("atomic_mass")
    features["avg_atomic_mass"] = weighted_avg("atomic_mass")
    features["max_atomic_mass"] = max(vals)
    features["min_atomic_mass"] = min(vals)
    features["range_atomic_mass"] = max(vals) - min(vals)
    features["std_atomic_mass"] = np.std(vals)

    return features
