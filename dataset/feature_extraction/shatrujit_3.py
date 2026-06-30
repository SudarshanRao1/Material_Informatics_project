import numpy as np
from pymatgen.core import Composition

def safe_get(el, prop):
    try:
        val = getattr(el, prop)
        return val if val is not None else None
    except:
        return None

def shatrujit_features_ratios(comp):
    comp = Composition(comp)
    elements = comp.elements
    fractions = np.array([comp.get_atomic_fraction(el) for el in elements])

    features = {}
# this code is classifing the compositions into different catogires
    metal_frac = sum(f for el, f in zip(elements, fractions) if safe_get(el, "is_metal"))
    heavy_frac = sum(f for el, f in zip(elements, fractions) if safe_get(el, "Z") and safe_get(el, "Z") > 50)
    transition_frac = sum(f for el, f in zip(elements, fractions) if safe_get(el, "is_transition_metal"))

    features["metal_fraction"] = metal_frac
    features["nonmetal_fraction"] = 1 - metal_frac
    features["heavy_element_fraction"] = heavy_frac
    features["light_element_fraction"] = 1 - heavy_frac
    features["transition_metal_fraction"] = transition_frac

    return features
