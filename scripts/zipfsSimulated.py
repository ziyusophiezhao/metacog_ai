# zipfsSimulated.py — Python 3 adaptation of Lydon-Staley et al. (2021)
# Fits a power law to concept frequency ranks and returns the alpha exponent.
# Values near 1 follow Zipf's law; higher alpha = steeper rank-frequency drop-off.

import numpy as np
import powerlaw
from collections import Counter


def zipfsSimulated(sequence: list) -> float:
    if len(sequence) < 2:
        return 0.0

    counts = np.array(list(Counter(sequence).values()), dtype=float)
    frequencies = counts[np.argsort(-counts)]

    if len(frequencies) < 2:
        return 0.0

    try:
        fit = powerlaw.Fit(frequencies, discrete=True, verbose=False)
        return float(fit.power_law.alpha)
    except (RuntimeError, ValueError):
        return 0.0
