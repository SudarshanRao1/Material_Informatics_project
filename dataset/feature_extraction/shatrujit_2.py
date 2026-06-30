import numpy as np
from pymatgen.core import Composition

def safe_get(el, prop):
    try:
        val = getattr(el, prop)
        return val if val is not None else None
    except:
        return None

def shatrujit_features_differences(comp):
    comp = Composition(comp)
    elements = comp.elements

    features = {}
# using a for loop the contrast features in the compositons were added
    X = []
    mass = []
    radius = []
    Z = []

    for el in elements:
        val_X = safe_get(el, "X")
        val_mass = safe_get(el, "atomic_mass")
        val_radius = safe_get(el, "atomic_radius")
        val_Z = safe_get(el, "Z")

        if val_X is not None:
            X.append(val_X)
        if val_mass is not None:
            mass.append(val_mass)
        if val_radius is not None:
            radius.append(val_radius)
        if val_Z is not None:
            Z.append(val_Z)

    features["electronegativity_diff"] = max(X) - min(X) if X else 0
    features["atomic_mass_diff"] = max(mass) - min(mass) if mass else 0
    features["atomic_radius_diff"] = max(radius) - min(radius) if radius else 0
    features["Z_diff"] = max(Z) - min(Z) if Z else 0
    features["radius_std"] = np.std(radius) if radius else 0

    return features
