# testFit.py — Python 3 adaptation of Lydon-Staley et al. (2021)
# Evaluates optimized NSGA-II parameters on held-out test folds.
#
# Usage (single fold):
#   python scripts/testFit.py --fold output/kFolds/<pid>_fold1_Test.txt
#                             --network output/underlyingNetworks/<pid>.csv
#                             --logbook output/kFolds/<pid>_fold1_logbook.pickle
#
# Or use 04_run_all.py to run all participants automatically.

from __future__ import division
import argparse
import glob
import pickle
import sys
from pathlib import Path

import networkx as nx
import numpy as np
from deap import base, creator

sys.path.insert(0, str(Path(__file__).parent))
from heapsSimulated import heapsSimulated
from entropySimulated import entropySimulated
from zipfsSimulated import zipfsSimulated
from intervalsSimulated import intervalsSimulated
from errwLevyFunction import generateLevy

# ── Argument parsing ──────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--fold",    type=str, default=None)
parser.add_argument("--network", type=str, default=None)
parser.add_argument("--logbook", type=str, default=None)
parser.add_argument("--subject", type=int, default=0)
args = parser.parse_args()

if args.fold and args.network and args.logbook:
    fold_path    = args.fold
    network_path = args.network
    logbook_path = args.logbook
else:
    test_list    = sorted(glob.glob("output/kFolds/*_Test.txt"))
    network_list = sorted(glob.glob("output/underlyingNetworks/*.csv"))
    logbook_list = sorted(glob.glob("output/kFolds/*_logbook.pickle"))

    if not test_list or not logbook_list:
        sys.exit("Missing test folds or logbooks. Run nsga.py first.")

    fold_path    = test_list[args.subject]
    pid          = Path(fold_path).stem.rsplit("_fold", 1)[0]
    network_path = [p for p in network_list if Path(p).stem == pid][0]
    logbook_path = logbook_list[args.subject]

print(f"Test fold:  {fold_path}")
print(f"Network:    {network_path}")
print(f"Logbook:    {logbook_path}")

# ── Load data ─────────────────────────────────────────────────────────────────

empirical_seq = np.loadtxt(fold_path).astype(int).tolist()
seq_len = len(empirical_seq)

heaps_emp     = heapsSimulated(empirical_seq)
entropy_emp   = entropySimulated(empirical_seq)
zipfs_emp     = zipfsSimulated(empirical_seq)
intervals_emp = intervalsSimulated(empirical_seq)

adj    = np.genfromtxt(network_path, delimiter=",")
G_base = nx.from_numpy_array(adj)
if not nx.is_connected(G_base):
    lcc    = max(nx.connected_components(G_base), key=len)
    G_base = G_base.subgraph(lcc).copy()
diam = nx.diameter(nx.minimum_spanning_tree(G_base))

# ── Extract best individual from logbook ─────────────────────────────────────

if "FitnessMin" not in creator.__dict__:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0))
if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMin)

with open(logbook_path, "rb") as f:
    logbook = pickle.load(f)

final_pop = logbook[-1]["pop"]
# Best individual = minimum total cost across four objectives
best = min(final_pop, key=lambda ind: sum(ind.fitness.values))
reinforcement, levy = best[0], best[1]
print(f"\nBest parameters  reinforcement={reinforcement:.4f}  levy={levy:.4f}")

# ── Evaluate on test fold ─────────────────────────────────────────────────────

sim_seq = generateLevy(seq_len, reinforcement, levy, G_base.copy(), diam)

costs = {
    "heaps_cost":     (heapsSimulated(sim_seq)     - heaps_emp)     ** 2,
    "entropy_cost":   (entropySimulated(sim_seq)   - entropy_emp)   ** 2,
    "zipfs_cost":     (zipfsSimulated(sim_seq)     - zipfs_emp)     ** 2,
    "intervals_cost": (intervalsSimulated(sim_seq) - intervals_emp) ** 2,
    "reinforcement":  reinforcement,
    "levy":           levy,
    "pid":            Path(fold_path).stem.rsplit("_fold", 1)[0],
    "fold":           Path(fold_path).stem.rsplit("_fold", 1)[1].replace("_Test", ""),
}

total_cost = sum(v for k, v in costs.items() if k.endswith("_cost"))
costs["total_cost"] = total_cost
print(f"Test costs  {costs}")

out_path = fold_path.replace("_Test.txt", "_testCosts.pickle")
with open(out_path, "wb") as f:
    pickle.dump(costs, f)
print(f"Test costs saved → {out_path}")
