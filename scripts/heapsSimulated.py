# heapsSimulated.py — Python 3 adaptation of Lydon-Staley et al. (2021)
# Fits Heaps' law to a concept sequence and returns the growth exponent b.
# A higher b indicates faster vocabulary growth (broader exploration).

import numpy as np
from scipy.optimize import curve_fit


def heapsSimulated(sequence: list) -> float:
    def power_law(x, a, b):
        return a * np.power(x, b)

    all_nodes, total_len, unique_len = [], [], []
    for node in sequence:
        all_nodes.append(node)
        total_len.append(len(all_nodes))
        unique_len.append(len(set(all_nodes)))

    x = np.array(total_len, dtype=float)
    y = np.array(unique_len, dtype=float)

    try:
        popt, _ = curve_fit(power_law, x, y, maxfev=5000)
        return float(popt[1])
    except (RuntimeError, ValueError):
        return 0.0
