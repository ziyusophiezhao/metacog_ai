# intervalsSimulated.py — Python 3 adaptation of Lydon-Staley et al. (2021)
# Computes the distribution of inter-event intervals (gaps between concept
# revisitations) and fits a power law.  Alpha captures how quickly the
# probability of a long gap decays — lower alpha = longer memory.

import numpy as np
import powerlaw
from collections import Counter
from functools import partial


def _occurrences(seq: list, item) -> list[int]:
    """Return all indices where item appears in seq."""
    locs, start = [], -1
    while True:
        try:
            loc = seq.index(item, start + 1)
        except ValueError:
            break
        locs.append(loc)
        start = loc
    return locs


def intervalsSimulated(sequence: list) -> float:
    if not sequence:
        return 0.0

    seq = list(sequence)
    find = partial(_occurrences, seq)
    all_intervals = []

    for node in set(seq):
        idxs = find(node)
        all_intervals.extend(j - i for i, j in zip(idxs[:-1], idxs[1:]))

    if len(all_intervals) < 2:
        return 0.0

    counts = np.array(list(Counter(all_intervals).values()), dtype=float)
    frequencies = counts[np.argsort(-counts)]

    if len(frequencies) < 2:
        return 0.0

    try:
        fit = powerlaw.Fit(frequencies, discrete=True, verbose=False)
        return float(fit.power_law.alpha)
    except (RuntimeError, ValueError):
        return 0.0
