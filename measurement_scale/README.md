# AI Metacognition Project


var APP_URL   = "https://metacog-ai.onrender.com";
var MAX_TURNS = 5;
var SESSION   = 1;  // stays 1 — one session per daily survey

// Production (reads from Qualtrics embedded data):
var topic = Qualtrics.SurveyEngine.getEmbeddedData('topic') || "${e://Field/topic}";
```

---
## Measurement Scales

Three instruments are used: one administered once at baseline, two repeated after each daily LLM session.

| Scale | Type | When | Source |
|---|---|---|---|
| MCQ-30 | Trait metacognition | Baseline only | Wells & Cartwright-Hatton (2004) |
| Processing Fluency | State metacognition | Daily, post-chat | Shulman & Sweitzer (2018a, 2018b) |
| Perceived Knowledge | State confidence | Daily, post-chat | Shulman & Sweitzer (2018b) |

**MCQ-30** measures stable beliefs about thinking and worrying across five subscales (positive beliefs about worry, uncontrollability and danger, cognitive confidence, need to control thoughts, cognitive self-consciousness). Given once at baseline to account for individual differences.

**Processing Fluency** (3 items) asks how smoothly participants felt they processed the LLM's responses, whether the language felt difficult, how new the information felt, and how easily they could form opinions (two items reverse-coded).

**Perceived Knowledge** (adapted from Shulman & Sweitzer's Perceived Political Knowledge measure) asks how informed and confident participants feel about the session topic, both on the specific topic and more broadly. Items are adapted for each condition (`poli_conspiracy` / `selective_exposure`).

---



