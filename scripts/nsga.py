# nsga.py — Python 3 adaptation of Lydon-Staley et al. (2021)
# NSGA-II multi-objective evolutionary optimization of Lévy flight parameters.
#
# Fits two parameters per participant fold:
#   reinforcement  (0–100)  — edge weight growth per traversal
#   levy_coeff     (1–3)    — Lévy flight step-size distribution exponent
#
# Usage (single fold):
#   python scripts/nsga.py --fold output/kFolds/<pid>_fold1_Train.txt
#                          --network output/underlyingNetworks/<pid>.csv
#
# Or use 04_run_all.py to process all participants automatically.

from __future__ import division
import argparse
import copy
import glob
import pickle
import random
import sys
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import networkx as nx
import numpy as np
from deap import algorithms, base, creator, tools
import deap.tools

sys.path.insert(0, str(Path(__file__).parent))
from heapsSimulated import heapsSimulated
from entropySimulated import entropySimulated
from zipfsSimulated import zipfsSimulated
from intervalsSimulated import intervalsSimulated
from errwLevyFunction import generateLevy

# ── Argument parsing ──────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--fold",    type=str, default=None,
                    help="Path to training fold .txt file")
parser.add_argument("--network", type=str, default=None,
                    help="Path to adjacency matrix .csv file")
parser.add_argument("--subject", type=int, default=0,
                    help="Index into sorted fold list (legacy mode)")
args = parser.parse_args()

# Resolve paths either from explicit args or legacy index
if args.fold and args.network:
    fold_path    = args.fold
    network_path = args.network
else:
    fold_list    = sorted(glob.glob("output/kFolds/*_Train.txt"))
    network_list = sorted(glob.glob("output/underlyingNetworks/*.csv"))
    if not fold_list:
        sys.exit("No training fold files found in output/kFolds/. "
                 "Run 03_build_network.py first.")
    fold_path    = fold_list[args.subject]
    pid          = Path(fold_path).stem.rsplit("_fold", 1)[0]
    matches      = [p for p in network_list if Path(p).stem == pid]
    if not matches:
        sys.exit(f"No network file found for participant {pid}.")
    network_path = matches[0]

print(f"Fold:    {fold_path}")
print(f"Network: {network_path}")

# ── Load data ─────────────────────────────────────────────────────────────────

empirical_seq = np.loadtxt(fold_path).astype(int).tolist()
if len(empirical_seq) < 4:
    sys.exit(f"Sequence too short ({len(empirical_seq)} concepts) for "
             "meaningful optimization. Collect more sessions first.")

seq_len = len(empirical_seq)

# Compute empirical statistics once
heaps_emp     = heapsSimulated(empirical_seq)
entropy_emp   = entropySimulated(empirical_seq)
zipfs_emp     = zipfsSimulated(empirical_seq)
intervals_emp = intervalsSimulated(empirical_seq)

print(f"Empirical stats  heaps={heaps_emp:.4f}  entropy={entropy_emp:.4f}  "
      f"zipfs={zipfs_emp:.4f}  intervals={intervals_emp:.4f}")

# Load concept network
adj = np.genfromtxt(network_path, delimiter=",")
G_base = nx.from_numpy_array(adj)

if not nx.is_connected(G_base):
    lcc = max(nx.connected_components(G_base), key=len)
    G_base = G_base.subgraph(lcc).copy()
    print(f"Graph disconnected; using largest component "
          f"({G_base.number_of_nodes()} nodes).")

diam = nx.diameter(nx.minimum_spanning_tree(G_base))
print(f"Graph: {G_base.number_of_nodes()} nodes, "
      f"{G_base.number_of_edges()} edges, diameter={diam}")

# ── Objective functions ───────────────────────────────────────────────────────


def f1(seq): return (heapsSimulated(seq)     - heaps_emp)     ** 2
def f2(seq): return (entropySimulated(seq)   - entropy_emp)   ** 2
def f3(seq): return (zipfsSimulated(seq)     - zipfs_emp)     ** 2
def f4(seq): return (intervalsSimulated(seq) - intervals_emp) ** 2


def evaluate(individual):
    reinforcement, levy = individual[0], individual[1]
    sim_seq = generateLevy(seq_len, reinforcement, levy, G_base.copy(), diam)
    if not sim_seq:
        return 1e6, 1e6, 1e6, 1e6
    return f1(sim_seq), f2(sim_seq), f3(sim_seq), f4(sim_seq)


# ── DEAP setup ────────────────────────────────────────────────────────────────

IND_SIZE = 2
LOWER    = [0.0, 1.0]
UPPER    = [100.0, 3.0]
POP_SIZE = 50
MU       = 50
LAMBDA   = 50
NGEN     = 100
CXPB     = 0.8
MUTPB    = 0.2
ETA      = 10


def uniform_params(lower, upper, _):
    return [random.uniform(lo, hi) for lo, hi in zip(lower, upper)]


# Guard: creator classes are registered once per process
if "FitnessMin" not in creator.__dict__:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, -1.0, -1.0))
if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("params",     uniform_params, LOWER, UPPER, IND_SIZE)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.params)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate",   evaluate)
toolbox.register("mate",       deap.tools.cxSimulatedBinaryBounded,
                               eta=ETA, low=LOWER, up=UPPER)
toolbox.register("mutate",     deap.tools.mutPolynomialBounded,
                               eta=ETA, low=LOWER, up=UPPER, indpb=0.1)
toolbox.register("select",     tools.selNSGA2)


def main():
    pop    = toolbox.population(n=MU)
    stats  = tools.Statistics()
    stats.register("pop", copy.deepcopy)
    pop, logbook = algorithms.eaMuPlusLambda(
        pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats
    )
    return pop, logbook


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\nStarting NSGA-II  pop={POP_SIZE}  generations={NGEN} ...")
    t0 = datetime.now()

    with Pool() as pool:
        toolbox.register("map", pool.map)
        pop, logbook = main()

    elapsed = (datetime.now() - t0).total_seconds()
    print(f"Optimization complete in {elapsed:.1f}s")

    out_path = fold_path.replace("_Train.txt", "_logbook.pickle")
    with open(out_path, "wb") as f:
        pickle.dump(logbook, f)
    print(f"Logbook saved → {out_path}")
