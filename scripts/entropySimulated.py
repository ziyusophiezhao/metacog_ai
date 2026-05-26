# entropySimulated.py — Python 3 adaptation of Lydon-Staley et al. (2021)
# Computes Shannon entropy of concept frequency distribution.
# Higher entropy = more uniform/diverse concept usage.

import math
from collections import Counter

import numpy as np


def entropySimulated(sequence: list) -> float:
    if not sequence:
        return 0.0

    counts = np.array(list(Counter(sequence).values()), dtype=float)
    counts = counts[np.argsort(-counts)]
    total = counts.sum()
    if total == 0:
        return 0.0

    return float(sum(
        (f / total) * math.log(total / f, 2)
        for f in counts if f > 0
    ))
