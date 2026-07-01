"""
Thermoelectric Merit-ZT Dataset Feature Engineering
====================================================
Expands the 51-feature StarryData2.0-derived dataset to 120 features.
All new features are physically motivated for thermoelectric transport.

Author   : Sudarshan , jeevan , shatrujit
Dataset  : StarryData2.0 (experimental), augmented with pymatgen descriptors
Target   : ZT (merit-ZT) — dimensionless thermoelectric figure of merit

Physical categories added
--------------------------
  1. Atomic Radius & Volume           (6 features)
  2. Electronic Structure             (12 features)
  3. Thermochemical & Cohesive        (10 features)
  4. Elasticity & Vibrational         (8 features)
  5. Structural & Bonding Chemistry   (9 features)
  6. Composition Topology             (8 features)
  7. Temperature-Coupled & Derived    (16 features)
                                      ───────────
  Total new                           69 features
  Original                            51 features
  Final total                         120 features

Usage
-----
  python te_feature_engineering.py \
      --input  final_featured_ID_dataset_og.csv \
      --output te_dataset_120features.csv

Dependencies
------------
  pip install pymatgen pandas numpy tqdm
"""

import argparse
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from pymatgen.core import Composition, Element

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def safe_get(element: Element, data_key: str):
    """Return element.data[data_key], or NaN if missing/None."""
    val = element.data.get(data_key, None)
    return float(val) if val is not None else np.nan


def weighted_stat(elements, fractions, prop_fn):
    """
    Compute weighted average and standard deviation of a scalar property.
    prop_fn: callable(Element) -> float
    Returns (weighted_avg, weighted_std) — NaN if all values missing.
    """
    vals = np.array([prop_fn(el) for el in elements], dtype=float)
    fracs = np.array(fractions, dtype=float)

    # Drop NaN elements from both arrays consistently
    mask = ~np.isnan(vals)
    if mask.sum() == 0:
        return np.nan, np.nan, np.nan, np.nan

    v, f = vals[mask], fracs[mask]
    f = f / f.sum()  # re-normalise after dropping NaN

    w_avg = np.sum(f * v)
    w_std = np.sqrt(np.sum(f * (v - w_avg) ** 2))
    w_max = np.max(v)
    w_min = np.min(v)
    return w_avg, w_std, w_max, w_min


def fraction_flag(elements, fractions, flag_fn):
    """Composition-weighted fraction of elements satisfying a boolean flag."""
    total = sum(fr * flag_fn(el) for el, fr in zip(elements, fractions))
    return total / sum(fractions)


# ─────────────────────────────────────────────────────────────────────────────
# Per-composition feature extractor
# ─────────────────────────────────────────────────────────────────────────────

def extract_new_features(comp_str: str) -> dict:
    """
    Parse a composition string and compute all 69 new features.
    Returns a flat dict; NaN for any property that cannot be computed.
    """
    features = {}

    try:
        comp = Composition(comp_str)
    except Exception:
        # Return all-NaN dict if composition string is unparseable
        return {}

    elements = list(comp.elements)
    fractions = [comp.get_atomic_fraction(el) for el in elements]
    total_frac = sum(fractions)

    # ── CATEGORY 1: Atomic Radius & Volume ──────────────────────────────────

    # avg_atomic_radius: controls bond lengths → phonon group velocity
    ar_avg, ar_std, ar_max, ar_min = weighted_stat(
        elements, fractions,
        lambda el: float(el.atomic_radius) if el.atomic_radius else np.nan
    )
    features["avg_atomic_radius"] = ar_avg  # Å; mean bond-length proxy
    features["max_atomic_radius"] = ar_max  # drives largest lattice distortion
    features["min_atomic_radius"] = ar_min  # smallest atom in compound
    features["range_atomic_radius"] = (ar_max - ar_min) if not np.isnan(ar_max) else np.nan
    # Note: radius_std & atomic_radius_diff already exist; we add range & max/min

    # avg_molar_volume: proxy for unit cell volume → Grüneisen γ
    mv_avg, mv_std, _, _ = weighted_stat(
        elements, fractions,
        lambda el: safe_get(el, "Molar volume")
    )
    features["avg_molar_volume"] = mv_avg  # cm³/mol
    features["molar_volume_std"] = mv_std  # volume heterogeneity

    # ── CATEGORY 2: Electronic Structure ────────────────────────────────────

    # Electron affinity statistics
    ea_avg, ea_std, ea_max, ea_min = weighted_stat(
        elements, fractions,
        lambda el: el.electron_affinity if el.electron_affinity is not None else np.nan
    )
    features["avg_electron_affinity"] = ea_avg  # eV; band edge position proxy
    features["max_electron_affinity"] = ea_max
    features["min_electron_affinity"] = ea_min
    features["std_electron_affinity"] = ea_std  # charge transfer driving force

    # Mendeleev number statistics (IUPAC ordering = validated Mendeleev proxy)
    mn_avg, mn_std, _, _ = weighted_stat(
        elements, fractions,
        lambda el: safe_get(el, "Mendeleev no")
    )
    features["avg_mendeleev_number"] = mn_avg  # periodic-table chemical ordering
    features["mendeleev_std"] = mn_std  # chemical diversity

    # Valence electron concentration (VEC) — Hume-Rothery / band-filling rule
    # VEC = Σ(x_i * n_valence_i), summed over all elements
    vec_vals = []
    for el, fr in zip(elements, fractions):
        try:
            # valence returns (n_s, n_p, n_d, n_f) — sum them; NaN safe
            v = el.valence
            n_val = int(v[0]) + int(v[1]) if len(v) >= 2 else np.nan
        except Exception:
            n_val = np.nan
        vec_vals.append((fr, n_val))

    vec_valid = [(f, v) for f, v in vec_vals if not np.isnan(v)]
    features["valence_electron_concentration"] = (
        sum(f * v for f, v in vec_valid) / sum(f for f, _ in vec_valid)
        if vec_valid else np.nan
    )  # VEC: critical for predicting phase stability & band topology in TE

    # Total valence electrons per atom (differs from VEC: uses n_electrons total)
    features["total_valence_electrons_per_atom"] = (
        sum(fr * el.n_electrons for el, fr in zip(elements, fractions))
    )  # 8-electron (Zintl) / 18-electron (half-Heusler) rule check

    # Unfilled and filled valence orbital counts
    def count_unfilled(el):
        """Estimate unfilled orbitals from valence shell."""
        try:
            elec_struct = el.full_electronic_structure
            # last shell electrons
            last_n = max(orb[0] for orb in elec_struct)
            shell_els = sum(orb[2] for orb in elec_struct if orb[0] == last_n)
            capacity = 18  # max for d-block, conservative upper bound
            return max(0, capacity - shell_els)
        except Exception:
            return np.nan

    ufo_avg, ufo_std, _, _ = weighted_stat(elements, fractions, count_unfilled)
    features["avg_unfilled_orbitals"] = ufo_avg  # controls carrier effective mass

    # sp-electron fraction: fraction coming from s+p states (low effective mass)
    def sp_fraction(el):
        try:
            elec = el.full_electronic_structure
            sp = sum(orb[2] for orb in elec if orb[1] in ('s', 'p'))
            total = sum(orb[2] for orb in elec)
            return sp / total if total > 0 else np.nan
        except Exception:
            return np.nan

    sp_avg, _, _, _ = weighted_stat(elements, fractions, sp_fraction)
    features["sp_electron_fraction"] = sp_avg  # sp-band systems → light effective mass

    # Density of solid (proxy for polarizability; soft, heavy = high polarizability)
    dens_avg, dens_std, _, _ = weighted_stat(
        elements, fractions,
        lambda el: safe_get(el, "Density of solid")  # kg/m³
    )
    features["avg_density"] = dens_avg  # heavy materials → low κ_L (mass scattering)
    features["std_density"] = dens_std  # density heterogeneity

    # ── CATEGORY 3: Thermochemical & Cohesive Energy ────────────────────────

    # Melting point: bond strength proxy → Debye temperature → κ_L
    mp_avg, mp_std, mp_max, mp_min = weighted_stat(
        elements, fractions,
        lambda el: safe_get(el, "Melting point")  # K
    )
    features["avg_melting_point"] = mp_avg  # governs operating temperature
    features["std_melting_point"] = mp_std  # mismatch → mass-fluctuation phonon scattering

    # Boiling point: complementary bond-strength indicator
    bp_avg, bp_std, _, _ = weighted_stat(
        elements, fractions,
        lambda el: safe_get(el, "Boiling point")  # K
    )
    features["avg_boiling_point"] = bp_avg
    features["std_boiling_point"] = bp_std

    # Liquid range = Boiling point − Melting point (thermal stability window)
    features["avg_liquid_range"] = (
        (bp_avg - mp_avg) if not (np.isnan(bp_avg) or np.isnan(mp_avg)) else np.nan
    )  # wide liquid range → thermodynamic softness → anharmonicity

    # Brinell hardness: lattice stiffness proxy; hard materials → high κ_L
    bh_avg, bh_std, _, _ = weighted_stat(
        elements, fractions,
        lambda el: safe_get(el, "Brinell hardness")
    )
    features["avg_brinell_hardness"] = bh_avg
    features["std_brinell_hardness"] = bh_std

    # Electrical resistivity of elements (prior to alloying) — baseline mobility
    er_avg, er_std, _, _ = weighted_stat(
        elements, fractions,
        lambda el: safe_get(el, "Electrical resistivity")  # Ω·m
    )
    features["avg_electrical_resistivity_elem"] = er_avg
    features["std_electrical_resistivity_elem"] = er_std

    # ── CATEGORY 4: Elasticity & Vibrational ────────────────────────────────

    # Bulk modulus: κ_L ∝ B × v_s × l (Slack model)
    bm_avg, bm_std, _, _ = weighted_stat(
        elements, fractions,
        lambda el: safe_get(el, "Bulk modulus")  # GPa
    )
    features["avg_bulk_modulus"] = bm_avg
    features["std_bulk_modulus"] = bm_std  # modulus variance → force-constant disorder

    # Young's modulus: longitudinal stiffness
    ym_avg, ym_std, _, _ = weighted_stat(
        elements, fractions,
        lambda el: safe_get(el, "Youngs modulus")  # GPa
    )
    features["avg_youngs_modulus"] = ym_avg
    features["std_youngs_modulus"] = ym_std

    # Rigidity (shear) modulus
    rm_avg, rm_std, _, _ = weighted_stat(
        elements, fractions,
        lambda el: safe_get(el, "Rigidity modulus")  # GPa
    )
    features["avg_rigidity_modulus"] = rm_avg

    # Poisson's ratio = (3B − 2G) / (2(3B + G))  — estimated from B and G
    if not (np.isnan(bm_avg) or np.isnan(rm_avg)) and (3 * bm_avg + rm_avg) != 0:
        features["poissons_ratio_estimate"] = (3 * bm_avg - 2 * rm_avg) / (
                2 * (3 * bm_avg + rm_avg)
        )  # controls transverse vs longitudinal phonon ratio
    else:
        features["poissons_ratio_estimate"] = np.nan

    # Sound velocity (longitudinal, from element data): enters Slack κ_L model
    vs_avg, vs_std, _, _ = weighted_stat(
        elements, fractions,
        lambda el: safe_get(el, "Velocity of sound")  # m/s
    )
    features["avg_sound_velocity"] = vs_avg  # v_s in Slack: κ_L ∝ M·θ_D³·V^(1/3)/(γ²·n^(2/3)·T)
    features["std_sound_velocity"] = vs_std

    # Debye temperature estimate from sound velocity and molar volume
    # θ_D ≈ (ħ/k_B) * v_s * (6π²n/V)^(1/3);  use simple element-level approximation
    # Here we use the empirical relation θ_D ≈ 251 * v_s * (ρ/M)^(1/3) (Lindemann)
    if not (np.isnan(vs_avg) or np.isnan(dens_avg) or np.isnan(mv_avg)):
        rho = dens_avg  # kg/m³
        V = mv_avg * 1e-6  # m³/mol
        M_kg = sum(el.atomic_mass * fr for el, fr in zip(elements, fractions)) * 1.66e-27
        # Approximate Debye: θ_D ≈ (ħ/k_B) * v_s * (6π²/V_atom)^(1/3)
        N_A = 6.022e23
        V_atom = (mv_avg * 1e-6) / N_A  # m³ per atom
        theta_D = (1.0546e-34 / 1.38e-23) * vs_avg * (6 * np.pi ** 2 / V_atom) ** (1 / 3)
        features["debye_temperature_estimate"] = theta_D  # K — enters all κ_L models
    else:
        features["debye_temperature_estimate"] = np.nan

    # ── CATEGORY 5: Structural & Bonding Chemistry ──────────────────────────

    # Ionic character (Pauling): IC = 1 − exp(−0.25 * ΔX²) per bond pair
    # Approximate by pairwise combinations weighted by fractions
    if len(elements) > 1:
        ic_values, ic_weights = [], []
        for i in range(len(elements)):
            for j in range(i + 1, len(elements)):
                dX = abs(elements[i].X - elements[j].X)
                ic = 1.0 - np.exp(-0.25 * dX ** 2)
                w = fractions[i] * fractions[j]
                ic_values.append(ic)
                ic_weights.append(w)
        tot_w = sum(ic_weights)
        features["avg_ionic_character"] = (
            sum(v * w for v, w in zip(ic_values, ic_weights)) / tot_w
            if tot_w > 0 else np.nan
        )  # Pauling ionic character: high → higher Seebeck, lower mobility
        features["std_ionic_character"] = (
            np.sqrt(sum(w * (v - features["avg_ionic_character"]) ** 2
                        for v, w in zip(ic_values, ic_weights)) / tot_w)
            if tot_w > 0 else np.nan
        )
    else:
        features["avg_ionic_character"] = 0.0
        features["std_ionic_character"] = 0.0

    # Oxidation state statistics
    def common_ox(el):
        ox = el.common_oxidation_states
        return float(np.mean(ox)) if len(ox) > 0 else np.nan

    ox_avg, ox_std, ox_max, ox_min = weighted_stat(elements, fractions, common_ox)
    features["avg_oxidation_state"] = ox_avg  # formal charge; controls carrier density
    features["std_oxidation_state"] = ox_std  # charge disorder → ionized impurity scattering
    features["max_oxidation_state_compound"] = (
        max(el.max_oxidation_state for el in elements)
    )  # highest redox state in compound
    features["min_oxidation_state_compound"] = (
        min(el.min_oxidation_state for el in elements)
    )  # lowest (most reducing) state

    # ── CATEGORY 3 (continued): Cohesive energy ─────────────────────────────
    # Cohesive energy per atom (kJ/mol) — Kittel / CRC Handbook values
    # Directly controls phonon stiffness and Debye temperature
    COHESIVE_ENERGY = {
        'H': 218, 'He': 0.08, 'Li': 159, 'Be': 324, 'B': 565, 'C': 711, 'N': 473,
        'O': 249, 'F': 79, 'Ne': 2, 'Na': 108, 'Mg': 146, 'Al': 330, 'Si': 446,
        'P': 315, 'S': 277, 'Cl': 121, 'Ar': 7, 'K': 89, 'Ca': 177, 'Sc': 378,
        'Ti': 470, 'V': 512, 'Cr': 397, 'Mn': 281, 'Fe': 413, 'Co': 424, 'Ni': 428,
        'Cu': 337, 'Zn': 131, 'Ga': 272, 'Ge': 372, 'As': 302, 'Se': 227, 'Br': 112,
        'Kr': 11, 'Rb': 82, 'Sr': 164, 'Y': 422, 'Zr': 607, 'Nb': 725, 'Mo': 659,
        'Tc': 661, 'Ru': 649, 'Rh': 553, 'Pd': 378, 'Ag': 284, 'Cd': 112, 'In': 243,
        'Sn': 302, 'Sb': 262, 'Te': 197, 'I': 107, 'Xe': 16, 'Cs': 78, 'Ba': 183,
        'La': 432, 'Ce': 417, 'Pr': 357, 'Nd': 328, 'Sm': 207, 'Eu': 178, 'Gd': 398,
        'Tb': 391, 'Dy': 294, 'Ho': 301, 'Er': 317, 'Tm': 232, 'Yb': 152, 'Lu': 428,
        'Hf': 619, 'Ta': 782, 'W': 849, 'Re': 774, 'Os': 791, 'Ir': 669, 'Pt': 564,
        'Au': 368, 'Hg': 64, 'Tl': 182, 'Pb': 195, 'Bi': 210, 'Th': 598, 'U': 534,
    }
    ce_vals = [(COHESIVE_ENERGY.get(el.symbol), fr)
               for el, fr in zip(elements, fractions)]
    ce_valid = [(v, f) for v, f in ce_vals if v is not None]
    if ce_valid:
        tot_f = sum(f for _, f in ce_valid)
        avg_ce = sum(v * f for v, f in ce_valid) / tot_f
        std_ce = np.sqrt(sum(f * (v - avg_ce) ** 2 for v, f in ce_valid) / tot_f)
    else:
        avg_ce, std_ce = np.nan, np.nan
    features["avg_cohesive_energy"] = avg_ce  # kJ/mol; bond strength → κ_L
    features["std_cohesive_energy"] = std_ce  # heterogeneity → phonon scattering

    # ── CATEGORY 6: Composition Topology ────────────────────────────────────

    # Stoichiometry L2 norm: measures distance from equal-molar composition
    fracs_arr = np.array(fractions)
    fracs_arr /= fracs_arr.sum()
    features["stoichiometry_L2_norm"] = float(np.linalg.norm(fracs_arr))
    # High L2 → one element dominates (e.g., 0.95-doped); low L2 → equal-molar

    # Shannon diversity index (element diversity)
    features["element_diversity_index"] = float(
        -np.sum(fracs_arr * np.log(fracs_arr + 1e-12))
    )  # maximal entropy → high configurational disorder → strong phonon scattering

    # Electronegativity: max and min (you have avg, range, variance, diff — not max/min)
    features["max_electronegativity"] = max(el.X for el in elements)
    features["min_electronegativity"] = min(el.X for el in elements)

    # Chalcogen fraction (S, Se, Te, Po) — chalcogenides dominate TE materials
    features["chalcogen_fraction"] = fraction_flag(
        elements, fractions, lambda el: el.is_chalcogen
    )  # explicit flag for the most important TE chemical family

    # Pnictogen fraction (N, P, As, Sb, Bi) — another key TE family
    def is_pnictogen(el):
        return el.symbol in {"N", "P", "As", "Sb", "Bi"}

    features["pnictogen_fraction"] = fraction_flag(elements, fractions, is_pnictogen)

    # Lanthanide fraction: f-electron → strong spin-orbit → anomalous Seebeck
    features["lanthanide_fraction"] = fraction_flag(
        elements, fractions, lambda el: el.is_lanthanoid
    )

    # Post-transition metal fraction (Tl, Pb, Bi, Po, At, Ga, In, Sn, etc.)
    features["post_transition_metal_fraction"] = fraction_flag(
        elements, fractions, lambda el: el.is_post_transition_metal
    )

    # ── CATEGORY 7: Temperature-Coupled & Physically Derived ────────────────
    # NOTE: These features require Temperature, S, σ, κ values from the row.
    # They are computed as a vectorized step after all rows are processed.
    # (Placeholder — populated in the main pipeline below)

    return features


# ─────────────────────────────────────────────────────────────────────────────
# Temperature-coupled features (vectorised, applied after composition loop)
# ─────────────────────────────────────────────────────────────────────────────

LORENZ_NUMBER = 2.44e-8  # W·Ω·K⁻²  (Sommerfeld value)


def add_temperature_coupled_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 16 temperature-coupled and physics-derived features using existing
    transport columns: Temperature, Seebeck coefficient, Electrical conductivity,
    Thermal conductivity, ZT.
    """
    T = df["Temperature"].astype(float)
    S = df["Seebeck coefficient"].astype(float)  # µV/K — keep raw units
    S_SI = S * 1e-6  # V/K  — for ZT formula
    σ = df["Electrical conductivity"].astype(float)  # S/m
    κ = df["Thermal conductivity"].astype(float)  # W/(m·K)
    ZT = df["ZT"].astype(float)

    # 1. Temperature transforms
    df["T_squared"] = T ** 2  # non-linear T dependence
    df["log_T"] = np.log(T.clip(lower=1))  # Dulong-Petit high-T regime

    # 2. Homologous temperature (requires avg_melting_point from composition)
    if "avg_melting_point" in df.columns:
        df["T_over_avg_melting_point"] = T / df["avg_melting_point"].replace(0, np.nan)
        # T/T_melt: governs anharmonic expansion and rattling onset

    # 3. Reduced Debye temperature ratio
    if "debye_temperature_estimate" in df.columns:
        df["T_over_Debye"] = T / df["debye_temperature_estimate"].replace(0, np.nan)
        # T/θ_D: key phonon activation parameter in κ_L = A * θ_D³ / (γ²·n·T)

    # 4. Seebeck-based features
    df["abs_Seebeck"] = np.abs(S)  # |S|; sign encodes carrier type
    df["Seebeck_sign"] = np.sign(S).astype(float)  # +1 p-type / −1 n-type
    df["Seebeck_squared"] = S_SI ** 2  # S² — power factor numerator (V²/K²)

    # 5. Power factor proxy: S²σ (W/(m·K²))  — ZT numerator without T
    df["power_factor_proxy"] = (S_SI ** 2) * σ

    # 6. ZT numerator proxy: S²σT = κ·ZT  (should equal κ·ZT by definition)
    #    Computing independently as a cross-feature
    df["ZT_numerator_proxy"] = (S_SI ** 2) * σ * T  # W/m·K (= κ·ZT)

    # 7. σ/κ ratio — measures Wiedemann-Franz deviation; κ_e/κ_total
    df["sigma_over_kappa"] = σ / κ.replace(0, np.nan)

    # 8. Reduced ZT/T — removes explicit T scaling → exposes intrinsic material quality
    df["reduced_ZT_T"] = ZT / T.replace(0, np.nan)

    # 9. Log-transforms of conductivities (multi-decade ranges → more Gaussian)
    df["log_electrical_conductivity"] = np.log1p(σ.clip(lower=0))
    df["log_thermal_conductivity"] = np.log1p(κ.clip(lower=0))

    # 10. Electronic thermal conductivity (Wiedemann-Franz): κ_e = L·σ·T
    df["kappa_electronic_WF"] = LORENZ_NUMBER * σ * T  # W/(m·K)
    # This is the WF estimate; κ_lattice ≈ κ_total − κ_electronic

    # 11. Lattice thermal conductivity proxy
    df["kappa_lattice_proxy"] = (κ - df["kappa_electronic_WF"]).clip(lower=0)

    # 12. Temperature × avg_Z coupling
    if "avg_Z" in df.columns:
        df["T_times_avg_Z"] = T * df["avg_Z"]
        # Higher-Z elements at higher T → stronger Umklapp phonon scattering

    # 13. Temperature × VEC coupling
    if "valence_electron_concentration" in df.columns:
        df["T_times_VEC"] = T * df["valence_electron_concentration"]
        # Band-filling effects at temperature; relevant to bipolar conduction onset

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main(input_path: str, output_path: str):
    print(f"\n{'=' * 65}")
    print("  Thermoelectric Dataset Feature Engineering: 51 → 120")
    print(f"{'=' * 65}\n")

    # ── Load original dataset ────────────────────────────────────────────────
    print(f"[1/5] Loading dataset: {input_path}")
    df = pd.read_csv("/home/sudarshan/Documents/INFORMATICS/dataset/final_featured_ID_dataset_og.csv")
    print(f"      Original shape: {df.shape}")
    assert df.shape[1] == 51, (
        f"Expected 51 columns in input, found {df.shape[1]}. "
        "Check that you are using the correct base dataset."
    )

    # ── Extract composition-based features ──────────────────────────────────
    print("\n[2/5] Extracting composition-based features (may take ~2-3 min)...")
    comp_feature_list = []
    failed = []
    for idx, comp_str in enumerate(tqdm(df["composition"], desc="    Compositions")):
        try:
            feat = extract_new_features(str(comp_str))
        except Exception as e:
            feat = {}
            failed.append((idx, comp_str, str(e)))
        comp_feature_list.append(feat)

    if failed:
        print(f"\n  [WARNING] {len(failed)} compositions failed parsing:")
        for idx, c, err in failed[:10]:
            print(f"    row {idx}: '{c}' → {err}")

    comp_feat_df = pd.DataFrame(comp_feature_list, index=df.index)
    print(f"      Composition features extracted: {comp_feat_df.shape[1]} columns")

    # ── Merge with original ──────────────────────────────────────────────────
    print("\n[3/5] Merging with original dataset...")
    df_merged = pd.concat([df, comp_feat_df], axis=1)

    # ── Add temperature-coupled features ────────────────────────────────────
    print("\n[4/5] Adding temperature-coupled & physics-derived features...")
    df_final = add_temperature_coupled_features(df_merged)
    print(f"      Final shape: {df_final.shape}")

    # ── Quality report ───────────────────────────────────────────────────────
    print("\n[5/5] Quality report")
    print(f"  Samples  : {df_final.shape[0]}")
    print(f"  Features : {df_final.shape[1]}")

    new_cols = [c for c in df_final.columns if c not in df.columns]
    print(f"  New cols : {len(new_cols)}")

    null_report = df_final[new_cols].isnull().mean().sort_values(ascending=False)
    high_null = null_report[null_report > 0.10]
    if len(high_null) > 0:
        print(f"\n  Features with >10% missing (imputation recommended):")
        for col, pct in high_null.items():
            print(f"    {col:<45s}  {pct * 100:.1f}%")
    else:
        print("\n  All new features have ≤10% missing values ✓")

    # ── Impute modulus-related features using column median ─────────────────
    # (Ge, Ga, In, Si lack some modulus data in pymatgen — median is defensible
    #  for a compositional dataset since these are minority-element compositions)
    modulus_cols = [c for c in new_cols if "modulus" in c or "poissons" in c or "sound" in c]
    for col in modulus_cols:
        if df_final[col].isnull().any():
            med = df_final[col].median()
            n_filled = df_final[col].isnull().sum()
            df_final[col].fillna(med, inplace=True)
            print(f"  Median-imputed '{col}': {n_filled} values → {med:.4f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    df_final.to_csv(output_path, index=False)
    print(f"\n  Output saved → {output_path}")

    # ── Column manifest ──────────────────────────────────────────────────────
    print(f"\n{'─' * 65}")
    print("  FULL COLUMN MANIFEST")
    print(f"{'─' * 65}")
    for i, col in enumerate(df_final.columns):
        tag = " [NEW]" if col in new_cols else ""
        print(f"  {i + 1:>3}. {col}{tag}")

    print(f"\n{'=' * 65}")
    print(f"  Done. Final dataset: {df_final.shape[0]} samples × {df_final.shape[1]} features")
    print(f"{'=' * 65}\n")

    return df_final


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Expand thermoelectric dataset from 51 to 120 features."
    )
    parser.add_argument(
        "--input",
        default="final_featured_ID_dataset_og.csv",
        help="Path to the original 51-feature CSV dataset.",
    )
    parser.add_argument(
        "--output",
        default="te_dataset_120features.csv",
        help="Path for the output 120-feature CSV dataset.",
    )
    args = parser.parse_args()
    main(args.input, args.output)
