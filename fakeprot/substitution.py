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

# WAG conditional substitution probabilities (off-diagonal only; rows sum to 1 excluding self)
_WAG_RAW: dict[str, dict[str, float]] = {
    "A": {"R": 0.017297, "N": 0.014046, "D": 0.037629, "C": 0.006259, "Q": 0.019734,
          "E": 0.080258, "G": 0.147647, "H": 0.003251, "I": 0.009507, "L": 0.044917,
          "K": 0.052938, "M": 0.005647, "F": 0.005160, "P": 0.046501, "S": 0.237246,
          "T": 0.119128, "W": 0.000419, "Y": 0.004818, "V": 0.147599},
    "R": {"A": 0.042927, "N": 0.021229, "D": 0.010599, "C": 0.004090, "Q": 0.081390,
          "E": 0.031368, "G": 0.079323, "H": 0.025530, "I": 0.009633, "L": 0.071231,
          "K": 0.391164, "M": 0.005397, "F": 0.003305, "P": 0.028594, "S": 0.112863,
          "T": 0.041859, "W": 0.004783, "Y": 0.009515, "V": 0.025200},
    "N": {"A": 0.025350, "R": 0.015438, "D": 0.218905, "C": 0.001562, "Q": 0.027355,
          "E": 0.042874, "G": 0.098426, "H": 0.030118, "I": 0.016738, "L": 0.012921,
          "K": 0.142533, "M": 0.001216, "F": 0.002139, "P": 0.006182, "S": 0.232454,
          "T": 0.095119, "W": 0.000230, "Y": 0.017231, "V": 0.013207},
    "D": {"A": 0.066598, "R": 0.007558, "N": 0.214674, "C": 0.000375, "Q": 0.017542,
          "E": 0.363991, "G": 0.108146, "H": 0.010443, "I": 0.002244, "L": 0.011733,
          "K": 0.035752, "M": 0.000859, "F": 0.001438, "P": 0.016649, "S": 0.092483,
          "T": 0.026958, "W": 0.000514, "Y": 0.007554, "V": 0.014492},
    "C": {"A": 0.074375, "R": 0.019583, "N": 0.010286, "D": 0.002516, "Q": 0.006422,
          "E": 0.004216, "G": 0.089108, "H": 0.006615, "I": 0.020286, "L": 0.117686,
          "K": 0.013028, "M": 0.006848, "F": 0.025267, "P": 0.010688, "S": 0.273152,
          "T": 0.081049, "W": 0.006341, "Y": 0.028621, "V": 0.203914},
    "Q": {"A": 0.043478, "R": 0.072253, "N": 0.033394, "D": 0.021837, "C": 0.001191,
          "E": 0.260988, "G": 0.034166, "H": 0.037506, "I": 0.005058, "L": 0.091492,
          "K": 0.212469, "M": 0.008794, "F": 0.002456, "P": 0.028974, "S": 0.071180,
          "T": 0.047066, "W": 0.000689, "Y": 0.004372, "V": 0.022637},
    "E": {"A": 0.111358, "R": 0.017537, "N": 0.032962, "D": 0.285352, "C": 0.000492,
          "Q": 0.164361, "G": 0.057070, "H": 0.005359, "I": 0.005120, "L": 0.017109,
          "K": 0.138893, "M": 0.001950, "F": 0.001888, "P": 0.021018, "S": 0.049474,
          "T": 0.044272, "W": 0.000494, "Y": 0.003675, "V": 0.041615},
    "G": {"A": 0.279269, "R": 0.060455, "N": 0.103154, "D": 0.115575, "C": 0.014184,
          "Q": 0.029332, "E": 0.077799, "H": 0.004625, "I": 0.002764, "L": 0.013595,
          "K": 0.041890, "M": 0.002088, "F": 0.002345, "P": 0.015445, "S": 0.177753,
          "T": 0.026418, "W": 0.002008, "Y": 0.003912, "V": 0.027390},
    "H": {"A": 0.021315, "R": 0.067442, "N": 0.109410, "D": 0.038684, "C": 0.003650,
          "Q": 0.111607, "E": 0.025324, "G": 0.016030, "I": 0.010588, "L": 0.105671,
          "K": 0.099039, "M": 0.004795, "F": 0.029994, "P": 0.042934, "S": 0.101950,
          "T": 0.051895, "W": 0.001716, "Y": 0.139856, "V": 0.018102},
    "I": {"A": 0.014248, "R": 0.005818, "N": 0.013901, "D": 0.001900, "C": 0.002559,
          "Q": 0.003441, "E": 0.005531, "G": 0.002190, "H": 0.002421, "L": 0.297716,
          "K": 0.016491, "M": 0.022043, "F": 0.021152, "P": 0.003266, "S": 0.020540,
          "T": 0.070198, "W": 0.000699, "Y": 0.007135, "V": 0.488753},
    "L": {"A": 0.067127, "R": 0.042894, "N": 0.010700, "D": 0.009907, "C": 0.014801,
          "Q": 0.062061, "E": 0.018429, "G": 0.010742, "H": 0.024088, "I": 0.296859,
          "K": 0.023424, "M": 0.042537, "F": 0.071082, "P": 0.020186, "S": 0.037871,
          "T": 0.029207, "W": 0.003217, "Y": 0.011782, "V": 0.203088},
    "K": {"A": 0.073819, "R": 0.219781, "N": 0.110127, "D": 0.028168, "C": 0.001529,
          "Q": 0.134474, "E": 0.139588, "G": 0.030882, "H": 0.021065, "I": 0.015343,
          "L": 0.021856, "M": 0.005668, "F": 0.002209, "P": 0.018504, "S": 0.071404,
          "T": 0.078296, "W": 0.000485, "Y": 0.002824, "V": 0.023979},
    "M": {"A": 0.040954, "R": 0.015771, "N": 0.004885, "D": 0.003520, "C": 0.004179,
          "Q": 0.028945, "E": 0.010194, "G": 0.008005, "H": 0.005304, "I": 0.106654,
          "L": 0.206410, "K": 0.029479, "F": 0.045875, "P": 0.009889, "S": 0.060869,
          "T": 0.143097, "W": 0.002854, "Y": 0.014120, "V": 0.258997},
    "F": {"A": 0.026641, "R": 0.006875, "N": 0.006119, "D": 0.004194, "C": 0.010978,
          "Q": 0.005754, "E": 0.007024, "G": 0.006402, "H": 0.023620, "I": 0.072863,
          "L": 0.245569, "K": 0.008178, "M": 0.032660, "P": 0.012673, "S": 0.091513,
          "T": 0.024149, "W": 0.011673, "Y": 0.287473, "V": 0.115641},
    "P": {"A": 0.185892, "R": 0.046058, "N": 0.013693, "D": 0.037606, "C": 0.003596,
          "Q": 0.052571, "E": 0.060555, "G": 0.032644, "H": 0.026179, "I": 0.008710,
          "L": 0.053996, "K": 0.053049, "M": 0.005451, "F": 0.009812, "S": 0.249006,
          "T": 0.098648, "W": 0.001011, "Y": 0.009221, "V": 0.052303},
    "S": {"A": 0.222810, "R": 0.042709, "N": 0.120963, "D": 0.049074, "C": 0.021588,
          "Q": 0.030341, "E": 0.033487, "G": 0.088258, "H": 0.014604, "I": 0.012871,
          "L": 0.023799, "K": 0.048090, "M": 0.007883, "F": 0.016646, "P": 0.058499,
          "T": 0.180595, "W": 0.001266, "Y": 0.011336, "V": 0.015181},
    "T": {"A": 0.163589, "R": 0.023161, "N": 0.072375, "D": 0.020916, "C": 0.009366,
          "Q": 0.029335, "E": 0.043816, "G": 0.019179, "H": 0.010870, "I": 0.064316,
          "L": 0.026837, "K": 0.077105, "M": 0.027096, "F": 0.006423, "P": 0.033887,
          "S": 0.264064, "W": 0.000395, "Y": 0.005696, "V": 0.101574},
    "W": {"A": 0.016381, "R": 0.075354, "N": 0.004989, "D": 0.011347, "C": 0.020864,
          "Q": 0.012231, "E": 0.013926, "G": 0.041505, "H": 0.010230, "I": 0.018227,
          "L": 0.084173, "K": 0.013595, "M": 0.015388, "F": 0.088394, "P": 0.009884,
          "S": 0.052715, "T": 0.011241, "Y": 0.317114, "V": 0.182441},
    "Y": {"A": 0.029772, "R": 0.023692, "N": 0.059003, "D": 0.026376, "C": 0.014884,
          "Q": 0.012263, "E": 0.016369, "G": 0.012781, "H": 0.131827, "I": 0.029420,
          "L": 0.048718, "K": 0.012513, "M": 0.012032, "F": 0.344082, "P": 0.014254,
          "S": 0.074592, "T": 0.025636, "W": 0.050124, "V": 0.061662},
    "V": {"A": 0.163439, "R": 0.011244, "N": 0.008103, "D": 0.009067, "C": 0.019001,
          "Q": 0.011377, "E": 0.033212, "G": 0.016035, "H": 0.003057, "I": 0.361091,
          "L": 0.150475, "K": 0.019041, "M": 0.039546, "F": 0.024801, "P": 0.014488,
          "S": 0.017900, "T": 0.081906, "W": 0.005167, "Y": 0.011049},
}

# WAG matrix as a (20, 20) numpy array; diagonal is 0 (handled as 1 - mutation_rate elsewhere)
WAG_MATRIX: np.ndarray = np.zeros((20, 20))
for _i, _from in enumerate(AMINO_ACIDS):
    for _j, _to in enumerate(AMINO_ACIDS):
        if _from != _to:
            WAG_MATRIX[_i, _j] = _WAG_RAW[_from].get(_to, 0.0)
WAG_CDFS: np.ndarray = np.apply_along_axis(_cdf, 1, WAG_MATRIX)

# Mean substitution probability between physicochemical groups, normalized per source group
_pc_subs_raw: dict[str, dict[str, float]] = {}
for _group_a in PHYSICOCHEMICAL_GROUPS:
    _pc_subs_raw[_group_a] = {}
    for _group_b in PHYSICOCHEMICAL_GROUPS:
        if _group_b != _group_a:
            vals = [
                _WAG_RAW[x].get(y, 0.0)
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
