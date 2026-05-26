# errwLevyFunction.py — Python 3 adaptation of Lydon-Staley et al. (2021)
# Lévy flight with edge reinforcement on a NetworkX weighted graph.
#
# Parameters
# ----------
# a : int   — number of steps (= empirical sequence length)
# b : float — edge reinforcement delta (0–100)
# c : float — Lévy coefficient (1 < c ≤ 3)
# d : nx.Graph — weighted concept co-occurrence graph (will be copied internally)
# e : int   — graph diameter (from minimum spanning tree)
#
# Returns
# -------
# list of source nodes visited (same length as steps taken)

from __future__ import division
import random

import networkx as nx
import numpy as np


def generateLevy(a: int, b: float, c: float,
                 d: nx.Graph, e: int) -> list:
    G = d.copy()
    diam = max(e, 1)
    n_steps = a

    x = np.arange(1, diam + 1, dtype=float)
    pdf = np.power(1.0 / x, c)
    pdf /= pdf.sum()

    edges = list(G.edges)
    if not edges:
        return []

    start_edge = random.choice(edges)
    source_nodes = [start_edge[0]]
    target_nodes = []

    for k in range(n_steps):
        step_size = int(np.random.choice(x, p=pdf))
        path_sources = [source_nodes[k]]
        path_targets = []

        for step in range(step_size):
            current = path_sources[step]
            neighbors = list(G.neighbors(current))
            if not neighbors:
                break

            weights = np.array(
                [G[current][nb].get("weight", 1.0) for nb in neighbors],
                dtype=float
            )
            if weights.sum() == 0:
                weights = np.ones(len(weights))
            weights /= weights.sum()

            if len(neighbors) == 1:
                chosen = neighbors[0]
            else:
                chosen = np.random.choice(neighbors, p=weights)

            path_targets.append(chosen)
            path_sources.append(chosen)

            if step + 1 == step_size:
                origin, dest = path_sources[0], path_targets[-1]
                if G.has_edge(origin, dest):
                    G[origin][dest]["weight"] += b
                target_nodes.append(dest)
                if k + 1 < n_steps:
                    source_nodes.append(dest)

    return source_nodes
