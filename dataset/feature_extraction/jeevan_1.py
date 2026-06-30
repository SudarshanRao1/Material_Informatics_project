import numpy as np
from pymatgen.core import Composition

def safe_get(el, prop):
    try:
        val = getattr(el, prop)
        return val if val is not None else None
    except:
        return None

def jeevan_features_block(comp):
    comp = Composition(comp)
    elements = comp.elements
    fractions = [comp.get_atomic_fraction(el) for el in elements]

    features = {}
# we are adding the electronic architecture
    for block in ["s", "p", "d", "f"]:
        features[f"{block}_block_fraction"] = sum(
            f for el, f in zip(elements, fractions)
            if safe_get(el, "block") == block
        )
# Electronegativity Variance(sigma^2)
    X = []
    for el in elements:
        val = safe_get(el, "X")
        if val is not None:
            X.append(val)

    features["electronegativity_variance"] = np.var(X) if X else 0

    return features
