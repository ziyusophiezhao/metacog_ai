# 04_run_all.py
# Run from project root: C:\Users\joels\Documents\Claude_files\
#
# Orchestrates the full optimization pipeline for all participants and folds:
#   1. Runs NSGA-II (nsga.py) for each train fold
#   2. Evaluates best parameters on each test fold (testFit.py)
#   3. Aggregates test-cost results into output/processed/network_results.csv
#
# Each fold is processed sequentially to avoid memory pressure on a laptop GPU.
# Multiprocessing within each NSGA-II run is handled by nsga.py itself.

import glob
import pickle
import subprocess
import sys
from pathlib import Path

import pandas as pd

SCRIPTS = Path(__file__).parent

train_folds = sorted(glob.glob("output/kFolds/*_Train.txt"))
test_folds  = sorted(glob.glob("output/kFolds/*_Test.txt"))
networks    = sorted(glob.glob("output/underlyingNetworks/*.csv"))

if not train_folds:
    sys.exit("No training folds found. Run scripts 01–03 first.")

print(f"Found {len(train_folds)} training fold(s) across "
      f"{len(networks)} participant(s).\n")


def get_pid(path: str) -> str:
    return Path(path).stem.rsplit("_fold", 1)[0]


def get_network(pid: str) -> str | None:
    matches = [p for p in networks if Path(p).stem == pid]
    return matches[0] if matches else None


def get_logbook(train_path: str) -> str:
    return train_path.replace("_Train.txt", "_logbook.pickle")


def get_test_fold(train_path: str) -> str | None:
    candidate = train_path.replace("_Train.txt", "_Test.txt")
    return candidate if Path(candidate).exists() else None


# ── Step 1: Optimize all train folds ─────────────────────────────────────────

for i, train_path in enumerate(train_folds):
    pid = get_pid(train_path)
    net = get_network(pid)
    if net is None:
        print(f"[SKIP] No network for {pid}")
        continue

    logbook_path = get_logbook(train_path)
    if Path(logbook_path).exists():
        print(f"[SKIP] Logbook already exists: {logbook_path}")
        continue

    print(f"[{i+1}/{len(train_folds)}] Optimizing {Path(train_path).name} ...")
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "nsga.py"),
         "--fold", train_path, "--network", net],
        capture_output=False
    )
    if result.returncode != 0:
        print(f"  WARNING: nsga.py exited with code {result.returncode}")

# ── Step 2: Evaluate test folds ───────────────────────────────────────────────

for i, train_path in enumerate(train_folds):
    pid       = get_pid(train_path)
    net       = get_network(pid)
    test_path = get_test_fold(train_path)
    logbook   = get_logbook(train_path)

    if net is None or test_path is None:
        continue
    if not Path(logbook).exists():
        print(f"[SKIP] No logbook for {Path(train_path).name}")
        continue

    costs_path = test_path.replace("_Test.txt", "_testCosts.pickle")
    if Path(costs_path).exists():
        print(f"[SKIP] Test costs already exist: {costs_path}")
        continue

    print(f"[{i+1}/{len(train_folds)}] Evaluating {Path(test_path).name} ...")
    subprocess.run(
        [sys.executable, str(SCRIPTS / "testFit.py"),
         "--fold", test_path, "--network", net, "--logbook", logbook],
        capture_output=False
    )

# ── Step 3: Aggregate results ─────────────────────────────────────────────────

cost_files = sorted(glob.glob("output/kFolds/*_testCosts.pickle"))
if not cost_files:
    print("\nNo test cost files found. Optimization may still be running.")
    sys.exit(0)

rows = []
for cf in cost_files:
    with open(cf, "rb") as f:
        costs = pickle.load(f)
    rows.append(costs)

results_df = pd.DataFrame(rows)
results_df.to_csv("output/processed/network_results.csv", index=False)
print(f"\nAggregated results saved → output/processed/network_results.csv")
print(results_df.to_string(index=False))
