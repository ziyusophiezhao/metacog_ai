# 03_build_network.py
# Run from project root: C:\Users\joels\Documents\Claude_files\
#
# Builds per-participant concept co-occurrence networks and creates
# k-fold train/test splits of each participant's concept sequence.
#
# Network definition
# ------------------
# Nodes  : unique concept IDs (integers from global vocabulary)
# Edges  : two concepts are connected if they co-occur in the same session.
#          Non-adjacent sessions can share edges (a concept from session 1
#          and session 10 create an edge when they both appear in the same
#          session window).  Edge weight = number of sessions in which the
#          pair co-occurs.
#
# This mirrors the original paper's underlying Wikipedia network, where any
# two articles a participant visited in the same browsing session were linked.

import json
import sys
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np

# ── Load concept data ─────────────────────────────────────────────────────────

with open("output/processed/concept_vocabulary.json") as f:
    vocab: dict[str, int] = json.load(f)

with open("output/processed/participant_sessions_int.json") as f:
    participant_sessions: dict[str, dict[str, list[int]]] = json.load(f)

with open("output/processed/concept_sequences.json") as f:
    participant_sequences: dict[str, list[int]] = json.load(f)

n_concepts = len(vocab)
N_FOLDS = 3
MIN_SEQ_LEN = N_FOLDS * 2  # need at least 2 concepts per fold

Path("output/underlyingNetworks").mkdir(parents=True, exist_ok=True)
Path("output/kFolds").mkdir(parents=True, exist_ok=True)


# ── Network construction ──────────────────────────────────────────────────────

def build_cooccurrence_matrix(sessions_map: dict[str, list[int]],
                              n_nodes: int) -> np.ndarray:
    """
    Undirected weighted co-occurrence matrix.
    Edge(i, j) += 1 for each session in which both concept i and j appear.
    Non-adjacent sessions are allowed: any session can share an edge with any
    other session via shared concepts.
    """
    matrix = np.zeros((n_nodes, n_nodes), dtype=float)
    for concept_list in sessions_map.values():
        unique = list(set(concept_list))
        for c1, c2 in combinations(unique, 2):
            matrix[c1, c2] += 1.0
            matrix[c2, c1] += 1.0
    return matrix


def ensure_connectivity(matrix: np.ndarray) -> np.ndarray:
    """
    If the graph built from matrix is disconnected, add a small weight (0.01)
    between concepts that are adjacent in the flat concept sequence so the
    random walker always has somewhere to go.
    """
    G_check = nx.from_numpy_array(matrix)
    if nx.is_connected(G_check):
        return matrix
    # Add minimal sequential links across sessions to bridge components
    # (weight kept small so it doesn't dominate the co-occurrence signal)
    flat = [c for session in sorted(matrix.shape) for c in range(matrix.shape[0])]
    for i in range(len(flat) - 1):
        a, b = flat[i], flat[i + 1]
        if matrix[a, b] == 0:
            matrix[a, b] = 0.01
            matrix[b, a] = 0.01
    return matrix


# ── K-fold splitting ──────────────────────────────────────────────────────────

def create_kfolds(sequence: list[int], k: int = 3) -> list[list[int]]:
    """Split sequence into k roughly equal folds (temporal order preserved)."""
    n = len(sequence)
    fold_size = n // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else n
        folds.append(sequence[start:end])
    return folds


# ── Process each participant ──────────────────────────────────────────────────

for pid, sessions_map in participant_sessions.items():
    full_sequence = participant_sequences[pid]
    n_sessions = len(sessions_map)
    n_concepts_pid = len(set(full_sequence))

    print(f"\n{pid}  |  sessions={n_sessions}  concepts={n_concepts_pid}  "
          f"sequence_length={len(full_sequence)}")

    # Build co-occurrence matrix over the global vocabulary size
    matrix = build_cooccurrence_matrix(sessions_map, n_concepts)
    matrix = ensure_connectivity(matrix)

    # Save adjacency matrix
    out_net = f"output/underlyingNetworks/{pid}.csv"
    np.savetxt(out_net, matrix, delimiter=",", fmt="%.4f")
    print(f"  Network saved → {out_net}")

    # K-fold splits
    if len(full_sequence) < MIN_SEQ_LEN:
        print(f"  WARNING: sequence too short for {N_FOLDS}-fold CV "
              f"(length={len(full_sequence)}, need >={MIN_SEQ_LEN}).")
        print(f"  Saving full sequence as fold 1 train AND test (pilot mode).")
        np.savetxt(f"output/kFolds/{pid}_fold1_Train.txt",
                   full_sequence, fmt="%d")
        np.savetxt(f"output/kFolds/{pid}_fold1_Test.txt",
                   full_sequence, fmt="%d")
        continue

    folds = create_kfolds(full_sequence, k=N_FOLDS)
    for fold_idx, test_fold in enumerate(folds):
        train_folds = [f for i, f in enumerate(folds) if i != fold_idx]
        train_seq = [item for sublist in train_folds for item in sublist]

        np.savetxt(
            f"output/kFolds/{pid}_fold{fold_idx + 1}_Train.txt",
            train_seq, fmt="%d"
        )
        np.savetxt(
            f"output/kFolds/{pid}_fold{fold_idx + 1}_Test.txt",
            test_fold, fmt="%d"
        )
    print(f"  {N_FOLDS} folds saved → output/kFolds/{pid}_fold*.txt")

print("\nNetwork construction complete.")
print("Next: run nsga.py (or 04_run_all.py for all participants).")
