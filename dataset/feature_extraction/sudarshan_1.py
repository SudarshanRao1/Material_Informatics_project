import numpy as np
from pymatgen.core import Composition
# this entrire code adds the features related to atomic numbers of the composition
# a helper function where it will try to access any missing value if it is not found this function will return none
def safe_get(el, prop):
    try:
        val = getattr(el, prop)
        return val if val is not None else None
    except:
        return None
#turns strings into the smart object
def sudarshan_atomic_numbers(comp):
    comp = Composition(comp)
    elements = comp.elements
    fractions = [comp.get_atomic_fraction(el) for el in elements]
# this calculates the property based on the consentration
    def weighted_avg(prop):
        vals = []
        weights = []
        for el, f in zip(elements, fractions):
            val = safe_get(el, prop)
            if val is not None:
                vals.append(val * f)
                weights.append(f)
        return sum(vals) / sum(weights) if weights else 0
# makes an list of Z of all elements present in the composition
    def values(prop):
        vals = []
        for el in elements:
            val = safe_get(el, prop)
            if val is not None:
                vals.append(val)
        return vals if vals else [0.0]

    features = {}
#atomic numbers
    vals = values("Z")
    features["avg_Z"] = weighted_avg("Z")
    features["max_Z"] = max(vals)
    features["min_Z"] = min(vals)
    features["range_Z"] = max(vals) - min(vals)
    features["std_Z"] = np.std(vals)

    return features
