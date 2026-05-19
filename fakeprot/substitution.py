"""
WAG substitution model and amino acid physicochemical properties.

Source: Whelan, S. and N. Goldman. 2001. A general empirical model of protein
evolution derived from multiple protein families using a maximum-likelihood
approach. Molecular Biology and Evolution 18:691-699.
"""

import numpy as np

AMINO_ACIDS: list[str] = list("ARNDCQEGHILKMFPSTWYV")
AA_INDEX: dict[str, int] = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

# Background amino acid frequencies from WAG (2001), normalised to sum exactly to 1
_aa_freq_raw = np.array([
    0.086628, 0.043972, 0.039089, 0.057045, 0.019308,
    0.036728, 0.058059, 0.083252, 0.024431, 0.048466,
    0.086209, 0.062029, 0.019503, 0.038432, 0.045763,
    0.069518, 0.061013, 0.014386, 0.035274, 0.070896,
])
AA_FREQUENCIES: np.ndarray = _aa_freq_raw / _aa_freq_raw.sum()


def _cdf(probs: np.ndarray) -> np.ndarray:
    """Build a numerically stable cumulative distribution."""
    cdf = np.cumsum(probs, dtype=float)
    cdf[-1] = 1.0
    return cdf


AA_FREQUENCY_CDF: np.ndarray = _cdf(AA_FREQUENCIES)

# Physicochemical groups used to bias conservative amino acid substitutions
PHYSICOCHEMICAL_SETS: dict[str, list[str]] = {
    "Amide":                ["N", "Q"],
    "Aliphatic":            ["G", "A", "V", "L", "I"],
    "Basic":                ["H", "K", "R"],
    "With hydroxyl":        ["S", "T", "Y"],
    "With sulfur":          ["C", "M"],
    "Non-polar":            ["F", "G", "V", "L", "A", "I", "P", "M", "W"],
    "Polar":                ["Y", "S", "N", "T", "Q", "C"],
    "Very hydrophobic":     ["L", "I", "F", "W", "V", "M"],
    "Hydrophilic":          ["R", "K", "N", "Q", "P", "D"],
    "Positively charged":   ["K", "R"],
    "Negatively charged":   ["D", "E"],
    "Tiny":                 ["G", "A", "S"],
    "Small":                ["C", "D", "P", "N", "T"],
    "Median":               ["E", "V", "Q", "H"],
    "Big":                  ["M", "I", "L", "K", "R"],
    "Aromatic":             ["F", "Y", "W"],
    "Similar (Asn or Asp)": ["N", "D"],
    "Similar (Gln or Glu)": ["Q", "E"],
}

PHYSICOCHEMICAL_GROUPS: list[str] = list(PHYSICOCHEMICAL_SETS.keys())

# Boolean membership mask for each group over the 20 canonical AAs
PC_MASKS: dict[str, np.ndarray] = {
    group: np.array([aa in members for aa in AMINO_ACIDS])
    for group, members in PHYSICOCHEMICAL_SETS.items()
}

# Background frequencies for physicochemical groups (proportional to sum of member AA frequencies)
_pc_totals = np.array([AA_FREQUENCIES[PC_MASKS[g]].sum() for g in PHYSICOCHEMICAL_GROUPS])
PC_FREQUENCIES: np.ndarray = _pc_totals / _pc_totals.sum()

# WAG exchangeability parameters s_ij (symmetric, lower triangle, row-major, i > j)
# Order: ARNDCQEGHILKMFPSTWYV — same as AMINO_ACIDS above.
# Source: Whelan & Goldman (2001); values from PAML dat/wag.dat
_WAG_S_LOWER: list[float] = [
    # --- row 1  R ---
    0.551571,
    # --- row 2  N ---
    0.509848, 0.635346,
    # --- row 3  D ---
    0.738998, 0.147304, 5.429420,
    # --- row 4  C ---
    1.027040, 0.528191, 0.265256, 0.0302949,
    # --- row 5  Q ---
    0.908598, 3.035500, 1.543640, 0.616783, 0.0988179,
    # --- row 6  E ---
    1.582850, 0.439157, 0.947198, 6.174160, 0.021352, 5.469470,
    # --- row 7  G ---
    1.416720, 0.584665, 1.125560, 0.865584, 0.306674, 0.330052, 0.567717,
    # --- row 8  H ---
    0.316954, 2.137150, 3.956290, 0.930676, 0.248972, 4.294110, 0.570025, 0.249410,
    # --- row 9  I ---
    0.193335, 0.186979, 0.554236, 0.039437, 0.170135, 0.113917, 0.127395, 0.0304501, 0.138190,
    # --- row 10 L ---
    0.397915, 0.497671, 0.131528, 0.0848047, 0.384287, 0.869489, 0.154263, 0.0613037, 0.499462, 3.170970,
    # --- row 11 K ---
    0.906265, 5.351420, 3.012010, 0.479855, 0.0740339, 3.894900, 2.584430, 0.373558, 0.890432, 0.323832, 0.257555,
    # --- row 12 M ---
    0.893496, 0.683162, 0.198221, 0.103754, 0.390482, 1.545260, 0.315124, 0.174100, 0.404141, 4.257460, 4.854020, 0.934276,
    # --- row 13 F ---
    0.210494, 0.102711, 0.0961621, 0.0467304, 0.398020, 0.0999208, 0.0811339, 0.049931, 0.679371, 1.059470, 2.115170, 0.088836, 1.190630,
    # --- row 14 P ---
    1.438550, 0.679489, 0.195081, 0.423984, 0.109404, 0.933372, 0.682355, 0.243570, 0.696198, 0.0999288, 0.415844, 0.556896, 0.171329, 0.161444,
    # --- row 15 S ---
    3.370790, 1.224190, 3.974230, 1.071760, 1.407660, 1.028870, 0.704939, 1.341820, 0.740169, 0.319440, 0.344739, 0.967130, 0.493905, 0.545931, 1.613280,
    # --- row 16 T ---
    2.121110, 0.554413, 2.030060, 0.374866, 0.512984, 0.857928, 0.822765, 0.225833, 0.473307, 1.458160, 0.326622, 1.386980, 1.516120, 0.171903, 0.795384, 4.378020,
    # --- row 17 W ---
    0.113133, 1.163920, 0.0719167, 0.129767, 0.717070, 0.215737, 0.156557, 0.336983, 0.262569, 0.212483, 0.665309, 0.137505, 0.515706, 1.529640, 0.139405, 0.523742, 0.110864,
    # --- row 18 Y ---
    0.240735, 0.381533, 1.086000, 0.325711, 0.543833, 0.227710, 0.196303, 0.103604, 3.873440, 0.420170, 0.398618, 0.133264, 0.428437, 6.454280, 0.216046, 0.786993, 0.291148, 2.485390,
    # --- row 19 V ---
    2.006010, 0.251849, 0.196246, 0.152335, 1.002140, 0.301281, 0.588731, 0.187247, 0.118358, 7.821300, 1.800340, 0.305434, 2.058450, 0.649892, 0.314887, 0.232739, 1.388230, 0.365369, 0.314730,
]

# Build symmetric 20×20 exchangeability matrix S
_S_WAG = np.zeros((20, 20))
_k = 0
for _i in range(1, 20):
    for _j in range(_i):
        _S_WAG[_i, _j] = _S_WAG[_j, _i] = _WAG_S_LOWER[_k]
        _k += 1

# Q_ij = s_ij * π_j  (i ≠ j); diagonal set to 0 rather than to the negative row-sum,
# because we row-normalise immediately below to get the conditional substitution distribution.
# The diagonal value is irrelevant for sampling the target amino acid.
_Q_WAG = _S_WAG * AA_FREQUENCIES   # broadcast: column j scaled by π_j
np.fill_diagonal(_Q_WAG, 0.0)

# Row-normalise → P(j | i, substitution occurs) = q_ij / Σ_{k≠i} q_ik
WAG_MATRIX: np.ndarray = _Q_WAG / _Q_WAG.sum(axis=1, keepdims=True)
WAG_CDFS: np.ndarray = np.apply_along_axis(_cdf, 1, WAG_MATRIX)

# Mean substitution probability between physicochemical groups, normalized per source group
_pc_subs_raw: dict[str, dict[str, float]] = {}
for _group_a in PHYSICOCHEMICAL_GROUPS:
    _pc_subs_raw[_group_a] = {}
    for _group_b in PHYSICOCHEMICAL_GROUPS:
        if _group_b != _group_a:
            vals = [
                WAG_MATRIX[AA_INDEX[x], AA_INDEX[y]]
                for x in PHYSICOCHEMICAL_SETS[_group_a]
                for y in PHYSICOCHEMICAL_SETS[_group_b]
                if y != x
            ]
            _pc_subs_raw[_group_a][_group_b] = float(np.mean(vals)) if vals else 0.0

PC_SUBS_MATRIX: dict[str, dict[str, float]] = {
    a: {b: v / sum(_pc_subs_raw[a].values()) for b, v in _pc_subs_raw[a].items()}
    for a in _pc_subs_raw
}

PC_INDEX: dict[str, int] = {g: i for i, g in enumerate(PHYSICOCHEMICAL_GROUPS)}
CHAR_GAP: int = 20   # uint8 sentinel for a gap column
PC_NONE: int = -1    # int8 sentinel for no physicochemical group

PC_AA_CDFS: dict[str, np.ndarray] = {
    group: _cdf(AA_FREQUENCIES * mask / (AA_FREQUENCIES * mask).sum())
    for group, mask in PC_MASKS.items()
}

PC_WAG_CDFS: dict[str, np.ndarray] = {}
PC_WAG_TOTALS: dict[str, np.ndarray] = {}
for _group, _mask in PC_MASKS.items():
    _masked = WAG_MATRIX * _mask
    _totals = _masked.sum(axis=1)
    _normalised = np.zeros_like(_masked)
    _nonzero = _totals > 0.0
    _normalised[_nonzero] = _masked[_nonzero] / _totals[_nonzero, np.newaxis]
    PC_WAG_CDFS[_group] = np.apply_along_axis(_cdf, 1, _normalised)
    PC_WAG_TOTALS[_group] = _totals
