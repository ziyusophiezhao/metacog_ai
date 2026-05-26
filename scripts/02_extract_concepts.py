# 02_extract_concepts.py
# Run from project root: C:\Users\joels\Documents\Claude_files\
#
# Extracts named entities and key noun phrases from conversation text
# using spaCy rule-based NER, builds a concept vocabulary, and produces
# ordered concept sequences per participant across sessions.
#
# Requires: python -m spacy download en_core_web_lg

import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import spacy
from spacy.pipeline import EntityRuler

# ── Load spaCy model ──────────────────────────────────────────────────────────

print("Loading spaCy model...")
nlp = spacy.load("en_core_web_lg")

# Domain-specific patterns the standard NER often misses
DOMAIN_PATTERNS = [
    {"label": "CONCEPT", "pattern": "COVID-19"},
    {"label": "CONCEPT", "pattern": "covid-19"},
    {"label": "CONCEPT", "pattern": "coronavirus"},
    {"label": "CONCEPT", "pattern": "conspiracy theory"},
    {"label": "CONCEPT", "pattern": "conspiracy theories"},
    {"label": "CONCEPT", "pattern": "echo chamber"},
    {"label": "CONCEPT", "pattern": "echo chambers"},
    {"label": "CONCEPT", "pattern": "misinformation"},
    {"label": "CONCEPT", "pattern": "disinformation"},
    {"label": "CONCEPT", "pattern": "social media"},
    {"label": "CONCEPT", "pattern": "fake news"},
    {"label": "CONCEPT", "pattern": "media bias"},
    {"label": "CONCEPT", "pattern": "confirmation bias"},
    {"label": "CONCEPT", "pattern": "critical thinking"},
    {"label": "CONCEPT", "pattern": "public health"},
    {"label": "CONCEPT", "pattern": "herd immunity"},
    {"label": "CONCEPT", "pattern": "vaccine hesitancy"},
    {"label": "CONCEPT", "pattern": "mental health"},
    {"label": "CONCEPT", "pattern": "information literacy"},
    {"label": "CONCEPT", "pattern": "artificial intelligence"},
    {"label": "CONCEPT", "pattern": [{"LOWER": "ai"}]},
    {"label": "CONCEPT", "pattern": "large language model"},
    {"label": "CONCEPT", "pattern": "chatgpt"},
    {"label": "CONCEPT", "pattern": "ChatGPT"},
]

ruler = nlp.add_pipe("entity_ruler", before="ner")
ruler.add_patterns(DOMAIN_PATTERNS)

# Entity types to keep from spaCy's standard NER + custom CONCEPT label
KEEP_LABELS = {
    "PERSON", "ORG", "GPE", "NORP", "PRODUCT", "EVENT",
    "WORK_OF_ART", "LAW", "LOC", "CONCEPT"
}

# Generic words that slip through NER but carry no useful semantic content
STOPWORDS = {
    "thing", "things", "people", "person", "way", "lot", "time", "times",
    "kind", "type", "bit", "example", "information", "info", "topic",
    "question", "answer", "thank", "hello", "hi", "okay", "sure", "right",
    "yes", "no", "world", "point", "fact", "part", "place", "year",
    "today", "day", "week", "month", "number", "area", "side", "level",
}


def normalize(text: str) -> str:
    return text.lower().strip()


def extract_concepts(text: str) -> list[str]:
    """Return a list of normalized concept strings from one text block."""
    doc = nlp(text)
    concepts = []

    # Named entities (standard + custom domain patterns)
    for ent in doc.ents:
        if ent.label_ in KEEP_LABELS:
            concept = normalize(ent.text)
            if len(concept) >= 3 and concept not in STOPWORDS:
                concepts.append(concept)

    # Noun chunks as fallback for unlabeled domain concepts
    ent_spans = {(e.start, e.end) for e in doc.ents}
    for chunk in doc.noun_chunks:
        if (chunk.start, chunk.end) in ent_spans:
            continue  # already captured as named entity
        root = chunk.root
        if (
            root.pos_ in {"NOUN", "PROPN"}
            and not root.is_stop
            and len(root.lemma_) >= 4
            and root.lemma_.lower() not in STOPWORDS
        ):
            concepts.append(normalize(root.lemma_))

    return concepts


# ── Load cleaned session data ─────────────────────────────────────────────────

sessions = pd.read_csv("output/processed/sessions_clean.csv")
sessions["content"] = sessions["content"].fillna("")

# ── Extract concepts per participant per session ──────────────────────────────

# participant_sessions: {pid: {session_num: [concept_str, ...]}}
participant_sessions: dict[str, dict[int, list[str]]] = defaultdict(dict)

pids = sessions["pid"].unique()
print(f"Processing {len(pids)} participants...")

for pid in pids:
    pid_df = sessions[sessions["pid"] == pid]
    for session_num, sess_df in pid_df.groupby("session_num"):
        session_concepts: list[str] = []
        # Process turns in order; include both user and assistant content
        for _, row in sess_df.sort_values("turn").iterrows():
            concepts = extract_concepts(str(row["content"]))
            session_concepts.extend(concepts)
        participant_sessions[pid][int(session_num)] = session_concepts
    print(f"  {pid}: {sum(len(v) for v in participant_sessions[pid].values())} concepts "
          f"across {len(participant_sessions[pid])} session(s)")

# ── Build global concept vocabulary ──────────────────────────────────────────

all_concepts: set[str] = set()
for sessions_map in participant_sessions.values():
    for concepts in sessions_map.values():
        all_concepts.update(concepts)

vocab: dict[str, int] = {c: i for i, c in enumerate(sorted(all_concepts))}
print(f"\nGlobal vocabulary: {len(vocab)} unique concepts")

# ── Map text concepts → integer IDs ──────────────────────────────────────────

# participant_sessions_int: {pid: {session_num_str: [concept_id, ...]}}
participant_sessions_int: dict[str, dict[str, list[int]]] = {
    pid: {
        str(snum): [vocab[c] for c in concepts]
        for snum, concepts in sessions_map.items()
    }
    for pid, sessions_map in participant_sessions.items()
}

# Full ordered sequence per participant (session 1 → 2 → ... → 21)
participant_sequences: dict[str, list[int]] = {
    pid: [
        cid
        for snum in sorted(int(k) for k in smap.keys())
        for cid in smap[str(snum)]
    ]
    for pid, smap in participant_sessions_int.items()
}

# ── Save outputs ──────────────────────────────────────────────────────────────

Path("output/processed").mkdir(parents=True, exist_ok=True)

with open("output/processed/concept_vocabulary.json", "w") as f:
    json.dump(vocab, f, indent=2)

with open("output/processed/participant_sessions_int.json", "w") as f:
    json.dump(participant_sessions_int, f, indent=2)

with open("output/processed/participant_sessions_text.json", "w") as f:
    json.dump(
        {pid: {str(k): v for k, v in smap.items()}
         for pid, smap in participant_sessions.items()},
        f, indent=2
    )

with open("output/processed/concept_sequences.json", "w") as f:
    json.dump(participant_sequences, f, indent=2)

print("\nOutputs written to output/processed/")
print("Next: run 03_build_network.py")
