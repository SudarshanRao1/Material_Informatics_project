import numpy as np
from pymatgen.core import Composition

def safe_get(el, prop):
    try:
        val = getattr(el, prop)
        return val if val is not None else None
    except:
        return None

def shatrujit_features_composition(comp):
    comp = Composition(comp)
    elements = comp.elements
    fractions = np.array([comp.get_atomic_fraction(el) for el in elements])

    features = {}
# features of composition's complexity
    features["num_elements"] = len(elements)

    safe_frac = fractions + 1e-12
    features["mixing_entropy"] = -np.sum(safe_frac * np.log(safe_frac))

    features["max_fraction"] = np.max(fractions)
    features["min_fraction"] = np.min(fractions)
    features["fraction_std"] = np.std(fractions)

    return features
