# Metacognitive LLM Conversation Study — Analysis Pipeline

A computational pipeline for analyzing how conversations with large language models (LLMs) shape metacognitive experiences, confidence in learned information, and the semantic structure of knowledge acquired through AI-assisted dialogue.

This pipeline adapts the network-science methodology from [Lydon-Staley et al. (2021)](https://www.nature.com/articles/s41562-021-01119-1) — originally designed to model curiosity-driven Wikipedia navigation — to a longitudinal LLM conversation paradigm.

---

## Study Overview

Participants engage in 21 daily sessions with an LLM agent on a single assigned topic (information-seeking or political conspiracy). Each session consists of 5 conversational turns. After each session, participants complete brief surveys on their attitudes toward the AI, metacognitive experiences, and confidence in what they learned. A pre-study baseline captures trait-level metacognitive beliefs (MCQ-30) and general attitudes toward LLMs.

The overarching research question concerns a **Markov decision process** in which metacognitive experience and confidence at time *t* influence both the nature of the LLM conversation at *t+1* and subsequent metacognitive states — mapping the dynamics of AI-mediated knowledge construction over time.

---

## Repository Structure

```
.
├── data_files/
│   ├── metacog_sessions_pilotdata.csv       # Conversation logs + post-session surveys
│   └── metacog_ai_baseline_pilotdata.csv    # Pre-study baseline survey (Qualtrics export)
│
├── scripts/
│   ├── 01_preprocess.R                      # Data cleaning and survey composite scoring
│   ├── 02_extract_concepts.py               # NER-based concept extraction from conversations
│   ├── 03_build_network.py                  # Concept co-occurrence networks + k-fold splits
│   ├── heapsSimulated.py                    # Heaps' law: vocabulary growth exponent
│   ├── entropySimulated.py                  # Shannon entropy: concept diversity
│   ├── zipfsSimulated.py                    # Zipf's law: frequency rank distribution
│   ├── intervalsSimulated.py                # Inter-event intervals: revisitation timing
│   ├── errwLevyFunction.py                  # Lévy flight + edge reinforcement random walk
│   ├── nsga.py                              # NSGA-II multi-objective parameter optimization
│   ├── testFit.py                           # Parameter evaluation on held-out test folds
│   └── 04_run_all.py                        # End-to-end pipeline orchestrator
│
├── output/
│   ├── processed/                           # Cleaned data, concept sequences, survey scores
│   ├── underlyingNetworks/                  # Per-participant adjacency matrices (.csv)
│   ├── kFolds/                              # Train/test folds, logbooks, test costs
│   └── figures/                            # Visualizations (user-generated)
│
├── reference_articles/
│   ├── Lydon-Staley et al. - 2020 - article.pdf
│   └── Lydon-Staley et al. - 2020 - supplement.pdf
│
├── requirements.txt
└── README.md
```

---

## Prerequisites

### R
- R ≥ 4.2
- Packages: `tidyverse`

Install with:
```r
install.packages("tidyverse")
```

### Python
- Python ≥ 3.10
- PyCharm (or any environment with pip access)
- NVIDIA GPU optional (pipeline is CPU-bound; 3070 or equivalent speeds up multiprocessing)

Install dependencies:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

---

## Pipeline Execution

All scripts should be run from the **project root** directory (`Claude_files/`).

### Step 1 — Preprocess raw survey and session data (R)

```r
source("scripts/01_preprocess.R")
```

**What it does:**
- Cleans the Qualtrics session export (drops the Qualtrics question-text header row, removes the unnamed empty column, coerces survey items to numeric)
- Recodes out-of-range Q25 values (> 7 on a 1–7 scale) to `NA`
- Computes survey composites:
  - `AI_attitude_composite` — mean of 7 post-session AI attitude items (Q20, 1–7 scale)
  - `MetaCog_composite` — mean of 3 post-session metacognitive experience items (Q29, 1–7 scale)
  - `InfoConfidence` — post-session knowledge confidence slider (Q30, 0–100)
  - `MCQ_*` subscales — five MCQ-30 subscales (positive worry, uncontrollability, cognitive confidence, need for control, self-consciousness) from baseline
  - `LLM_affect`, `LLM_credibility`, `LLM_intelligence` — baseline LLM attitude composites (Q21–Q23)
- Merges session-level scores with baseline measures by participant ID
- **Outputs** to `output/processed/`: `sessions_clean.csv`, `baseline_clean.csv`, `survey_session_scores.csv`, `survey_merged.csv`

---

### Step 2 — Extract concepts from conversations (Python)

```bash
python scripts/02_extract_concepts.py
```

**What it does:**
- Loads cleaned conversation logs
- Applies spaCy (`en_core_web_lg`) named entity recognition across all turns (user + assistant) for each session
- Augments standard NER with domain-specific patterns (e.g., "COVID-19", "echo chamber", "misinformation", "ChatGPT") via a custom `EntityRuler`
- Retains entities of types: `PERSON`, `ORG`, `GPE`, `NORP`, `PRODUCT`, `EVENT`, `WORK_OF_ART`, `LAW`, `LOC`, and the custom `CONCEPT` label
- Supplements NER with noun chunk roots for unlabeled but content-rich terms
- Normalizes all concepts to lowercase and filters generic stopwords
- Builds a **global vocabulary** (concept string → integer ID) shared across all participants
- Produces an ordered concept sequence per participant (sessions in chronological order, turns in turn order within session)
- **Outputs** to `output/processed/`: `concept_vocabulary.json`, `participant_sessions_int.json`, `participant_sessions_text.json`, `concept_sequences.json`

---

### Step 3 — Build concept networks and k-fold splits (Python)

```bash
python scripts/03_build_network.py
```

**What it does:**
- Constructs a **weighted undirected co-occurrence network** per participant:
  - **Nodes**: unique concept IDs from the global vocabulary
  - **Edges**: two concepts are connected if they co-occur in the same session; **non-adjacent sessions are permitted** (a concept from session 1 and session 10 generate an edge if they share a session)
  - **Edge weight**: number of sessions in which the concept pair co-occurs
- Ensures graph connectivity (adds minimal bridging weights if components are disconnected)
- Saves each network as a square adjacency matrix to `output/underlyingNetworks/{pid}.csv`
- Creates **3-fold cross-validation splits** of each participant's concept sequence (temporal order preserved)
- **Outputs** to `output/kFolds/`: `{pid}_fold{k}_Train.txt`, `{pid}_fold{k}_Test.txt`
- Participants with sequences too short for 3-fold CV are flagged; their full sequence is saved as a single fold (pilot mode)

---

### Step 4 — Optimize and evaluate (Python)

**Run all participants automatically:**
```bash
python scripts/04_run_all.py
```

This orchestrates Steps 4a and 4b below for every participant and fold, then aggregates results into `output/processed/network_results.csv`.

**Or run individual folds manually:**

#### Step 4a — NSGA-II parameter optimization (training folds)
```bash
python scripts/nsga.py --fold output/kFolds/{pid}_fold1_Train.txt \
                       --network output/underlyingNetworks/{pid}.csv
```

Fits two parameters to each participant's training sequence using multi-objective evolutionary optimization (NSGA-II via [DEAP](https://github.com/deap/deap)):

| Parameter | Range | Interpretation |
|---|---|---|
| `reinforcement` | 0–100 | How strongly traversed concept edges are up-weighted (exploitation) |
| `levy_coeff` | 1–3 | Lévy flight exponent controlling step-size distribution (exploration) |

Four objective functions are minimized simultaneously — squared differences between simulated and empirical values of Heaps' law exponent, Shannon entropy, Zipf's law exponent, and inter-event interval distribution. Outputs a logbook pickle to `output/kFolds/`.

#### Step 4b — Test fold evaluation
```bash
python scripts/testFit.py --fold output/kFolds/{pid}_fold1_Test.txt \
                          --network output/underlyingNetworks/{pid}.csv \
                          --logbook output/kFolds/{pid}_fold1_logbook.pickle
```

Extracts the best-fitting individual from the Pareto front, evaluates it on the held-out test sequence, and saves cost scores and fitted parameters to `output/kFolds/`.

---

## Network Analysis Metrics

| Metric | Script | What it captures |
|---|---|---|
| **Heaps' law exponent** (β) | `heapsSimulated.py` | Rate of new concept acquisition — higher β = broader, more exploratory knowledge growth |
| **Shannon entropy** | `entropySimulated.py` | Diversity of concept usage — higher entropy = more evenly distributed engagement across concepts |
| **Zipf's law exponent** (α) | `zipfsSimulated.py` | Concentration of concept use — lower α = a few concepts dominate discussion |
| **Inter-event interval** (α_IEI) | `intervalsSimulated.py` | Return timing to revisited concepts — lower α = longer memory, more persistent engagement with specific ideas |
| **Reinforcement parameter** | `nsga.py` / `testFit.py` | Exploitation tendency — how strongly participants return to previously discussed concepts |
| **Lévy coefficient** | `nsga.py` / `testFit.py` | Exploration tendency — how far conceptual "jumps" tend to be in a single conversational move |

---

## Adaptations from Lydon-Staley et al. (2021)

| Dimension | Original paper | This pipeline |
|---|---|---|
| **Navigation medium** | Wikipedia hyperlink clicks | LLM conversation turns |
| **Network nodes** | Wikipedia article IDs | Named entities + domain concepts (spaCy NER) |
| **Network edges** | Within-session page co-navigation | Within-session concept co-occurrence (non-adjacent sessions allowed) |
| **Curiosity analogue** | Curiosity state scale | AI attitude composite, metacognitive experience composite, information confidence |
| **Parallelism** | SCOOP (HPC cluster) | Python `multiprocessing` (local) |
| **Language** | Python 2 | Python 3 ≥ 3.10 |

Non-adjacent edges are used (rather than restricting edges to consecutive sessions) because the Lévy flight + edge reinforcement model requires branching choices at each node to produce identifiable parameters. A strict temporal chain of 21 nodes offers the random walker no meaningful alternatives.

---

## Survey Measures

### Post-session (collected after each of the 21 sessions)
| Variable | Items | Scale | Composite |
|---|---|---|---|
| AI Responsiveness | Q20.1 | 1–7 | — |
| AI Source Reliability | Q20.2 | 1–7 | — |
| AI Agreeableness | Q20.3 | 1–7 | — |
| AI Ethics | Q20.4 | 1–7 | — |
| AI Understanding | Q20.5 | 1–7 | — |
| AI Satisfaction | Q20.6 | 1–7 | — |
| AI Future Use | Q20.7 | 1–7 | — |
| **AI Attitude Composite** | Q20.1–7 | 1–7 | Row mean |
| MetaCognitive Experience 1–3 | Q29.1–3 | 1–7 | Row mean |
| Information Confidence | Q30.1 | 0–100 slider | — |

### Baseline (pre-study, collected once)
| Scale | Items | Notes |
|---|---|---|
| MCQ-30 | 30 items, 1–4 | 5 subscales: positive worry, uncontrollability, cognitive confidence, need for control, self-consciousness |
| LLM Affect | Q21, 4 items, 1–7 | Likable, Pleasant, Appealing, Not Irritating |
| LLM Credibility | Q22, 6 items, 1–7 | Credible, Trustworthy, Honest, Reliable, Unbiased, Sincere |
| LLM Intelligence | Q23, 3 items, 1–7 | Intelligent, Smart, Capable |
| ChatGPT Quality | Q25, 12 items, 1–7 | Response quality assessment |

---

## Notes on Pilot Data

The included pilot dataset contains 8 participants, each with a single session. This is sufficient to verify the end-to-end pipeline but not to run meaningful cross-validation or NSGA-II optimization (sequences are too short). When sequences fall below the minimum length for 3-fold CV, the pipeline automatically saves the full sequence as a single fold and skips optimization with a warning. The full 21-session dataset will run through the pipeline unchanged.

---

## Citation

If you use or adapt this pipeline, please cite the original methodology:

> Lydon-Staley, D. M., Zhou, D., Blevins, A. S., Zurn, P., & Bassett, D. S. (2021). Hunters, busybodies, and the knowledge network building associated with curiosity. *Nature Human Behaviour*, *5*(3), 327–336. https://doi.org/10.1038/s41562-021-01119-1

---

## License

This repository is intended for academic research purposes. Please contact the authors before reusing data or survey instruments.
