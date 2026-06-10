# AI Metacognition Project

A longitudinal 21-day experiment platform that embeds an OpenAI LLM conversation
inside a Qualtrics survey. Participants complete a **baseline survey on Day 1**,
then a **daily follow-up survey** for 21 days. Each daily session includes a
topic-locked LLM conversation and metacognitive outcome measures.

---

## Pipeline

```
Baseline Survey (Day 1 only)
  Demographics · AI Measures · MCQ-30 · Topic randomization
  ↓
Daily Follow-up Survey (Days 1–21)
  Embedded LLM Chat (JavaScript inside Qualtrics block)
  ↓  calls https://metacog-ai.onrender.com/chat
  Flask App (pipeline.py) — OpenAI API
  ↓  logs transcript to SQLite
  Metacognitive Experience · Confidence Scale
```

---

## File Structure

```
metacog_ai/
├── pipeline.py
├── scipts
├── measurement_scale
├── knowledge_analysis_pipeline
├── requirements.txt
├── templates/
│   └── chat.html
└── README.md
```

---

## Current Configuration

| Setting | Value |
|---|---|
| Model | `gpt-4o-mini` |
| Max turns per session | `5` |
| Topic conditions | `poli_conspiracy`, `info_seeking` |
| Study design | Longitudinal, 21 days |
| Baseline survey | `SV_71mW9DfPrudBg0K` |
| Daily follow-up survey | `SV_cNfsfrFP8ofRc6W` |

---

## Topic Conditions

Participants are randomly assigned to one of two topic conditions at the
baseline survey (Day 1) and remain in that condition for all 21 daily sessions:

| Condition | Topic | Description |
|---|---|---|
| `poli_conspiracy` | Political Conspiracies | Controversial — COVID-19 conspiracy beliefs, misinformation, epistemic trust (Pew Research Center, 2020) |
| `info_seeking` | Psychological Aspects of Information Seeking | Non-controversial — curiosity, knowledge networks, learning behavior (Lydon-Staley et al., 2021) |

---

## Survey Structure

### Baseline Survey — Day 1 only
**Link:** https://ucdavis.co1.qualtrics.com/jfe/form/SV_71mW9DfPrudBg0K

```
Set Embedded Data
  topic = (empty, set by randomizer)
↓
Randomizer (50/50, evenly distributed)
  ├── topic = poli_conspiracy
  └── topic = info_seeking
↓
ID/Student No.
Consent form
Demographic (5 questions)
AI Measures (4 questions)
MCQ-30 (Wells & Cartwright-Hatton, 2004)
↓
Branch: IF topic = poli_conspiracy
  └── Topic intro: poli_conspiracy
Branch: IF topic = info_seeking
  └── Topic intro: info_seeking
↓
Metacognitive Experience (Shulman & Sweitzer, 2018a, 2018b)
↓
Confidence Scale
↓
End of Survey
```

### Daily Follow-up Survey — Days 2–21
**Link:** https://ucdavis.co1.qualtrics.com/jfe/form/SV_cNfsfrFP8ofRc6W

```
Set Embedded Data
  session_num = 1  ← increment each wave (2–21)
  topic = (empty, carried from baseline via participant panel)
↓
Branch: IF topic = poli_conspiracy
  └── Topic intro: poli_conspiracy
  └── LLM_Chatbot (5 exchanges, topic = poli_conspiracy)
Branch: IF topic = info_seeking
  └── Topic intro: info_seeking
  └── LLM_Chatbot (5 exchanges, topic = info_seeking)
↓
Metacognitive Experience (Shulman & Sweitzer, 2018a, 2018b)
↓
Confidence Scale
↓
End of Survey
```

---

## Longitudinal Follow-up Procedure (Days 2–21)

After completing the baseline survey on Day 1, participants are contacted
daily for 20 subsequent days via Qualtrics email distribution.

### How topic condition carries forward

At the end of the baseline survey, each participant's assigned `topic`
value (`poli_conspiracy` or `info_seeking`) is saved in their Qualtrics
response record. Before distributing the daily follow-up survey, export
the baseline response data and create a **Qualtrics Contact List** with
two columns:

| Column | Value |
|---|---|
| `Email` | Participant email address |
| `topic` | Their assigned condition from baseline (`poli_conspiracy` or `info_seeking`) |

Upload this contact list to Qualtrics under **Contacts → Create Contact List**.
When you distribute the daily follow-up survey using this contact list,
Qualtrics automatically pipes `topic` into each participant's session as
an embedded data field — no manual entry needed.

### Daily distribution setup

1. Go to **Distributions → Emails** in the daily follow-up survey
   (`SV_cNfsfrFP8ofRc6W`)
2. Click **Compose Email**
3. Under **To**, select your contact list
4. Set the **Send Date** for each wave (Day 2 through Day 21)
   — schedule all emails at once using Qualtrics's scheduled send
5. In **Survey Flow**, confirm `topic` is declared as an Embedded Data
   field at the very top — Qualtrics will read it from the contact list
   automatically
6. Update `session_num` in the Embedded Data block for each wave:
   Day 2 = `2`, Day 3 = `3`, ... Day 21 = `21`

> **Note on session_num:** Because `session_num` is hardcoded in the
> Survey Flow rather than passed via contact list, we will need to
> update it manually in the Survey Flow before distributing each wave,
> or use a Qualtrics contact list attribute to pass it dynamically
> alongside `topic`.

### Email template

> Subject: [Chatbot Project] Day {N} — Your daily session is ready
>
> Hi,
>
> Your Day {N} session for the Chatbot Project is now available.
> Please complete it at your earliest convenience today.
>
> [Take Today's Survey] ← Qualtrics survey link button
>
> This should take approximately 10–15 minutes.
> Thank you for your continued participation.

### Participant retention

To encourage completion across all 21 days:
- We will send a reminder email 24 hours after the initial daily invite
  if the survey has not been completed
- We also consider a completion incentive tied to the number of daily
  sessions completed (e.g. bonus compensation for completing
  15 or more of 21 sessions)


---

## Qualtrics Embedded Data Fields

### Baseline survey
| Field | Value |
|---|---|
| `topic` | Left empty — filled by Randomizer |

### Daily follow-up survey
| Field | Value |
|---|---|
| `session_num` | Set manually per wave (`1` through `21`) |
| `topic` | Passed from baseline via Qualtrics contact list or URL parameter |

---

## LLM_Chatbot Block Setup

Each LLM_Chatbot block contains a Text/Graphic question with the following
JavaScript. `SESSION` and `topic` are the only variables that differ:

```javascript
var APP_URL   = "https://metacog-ai.onrender.com";
var MAX_TURNS = 5;
var SESSION   = 1;  // stays 1 — one session per daily survey

// Production (reads from Qualtrics embedded data):
var topic = Qualtrics.SurveyEngine.getEmbeddedData('topic') || "${e://Field/topic}";
```

### Participant instructions

**Condition: poli_conspiracy**
> A family member recently told you they believe COVID-19 was intentionally
> created and released by powerful people — a view shared by roughly 1 in 4
> Americans (Pew Research Center, 2020). You want to learn more before your
> next conversation with them. Talk with the AI to gather information about
> this conspiracy theory — what people believe, why they believe it, and what
> the evidence says. You must complete all 5 exchanges before moving on.

**Condition: info_seeking**
> A classmate mentioned they always seem to know a little about everything,
> while you tend to go deep on a few topics you care about. You started
> wondering why people seek information so differently. Talk with the AI to
> learn more about the psychology of information seeking — why people are
> curious, how they explore topics, and how they build knowledge over time.
> You must complete all 5 exchanges before moving on.

---

## Deployment

App is deployed on [Render.com](https://render.com):

```
https://metacog-ai.onrender.com
```

Environment variables set on Render:

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key |
| `EXPORT_TOKEN` | Secret token to protect /export endpoint |

Start command:
```
gunicorn pipeline:app
```

---

## Data Export

Download all LLM conversation logs as CSV:

```
https://metacog-ai.onrender.com/export?token=your-export-token
```

CSV columns: `session_id`, `pid`, `topic`, `session_num`, `role`, `content`, `turn`, `timestamp`

Merge with Qualtrics exports using `pid` + `session_num` as the join key.

OR

Download directly from Qualtrics.

---

## Knowledge Network Analysis

Following Lydon-Staley et al. (2021), knowledge networks are built from
human–LLM chat logs. For each participant:

- Each complete exchange (user question + assistant response) is defined as a node
- The chronological sequence of exchanges across 21 sessions forms a traversal path
- Pairwise cosine similarities between node embeddings define edge weights
- Graph-theoretical indices are extracted: average edge weight, weighted clustering
  coefficient, characteristic path length, small-world propensity, modularity Q,
  and number of communities
- The sequence is split into early, middle, and late thirds for within-person
  time-varying analyses (ICC and repeated-measures correlations with
  metacognitive and confidence scores)

---

## Dependencies

```
openai
flask
flask-cors
gunicorn
```

---

## Key References

Lydon-Staley, D. M., Zhou, D., Blevins, A. S., Zurn, P., & Bassett, D. S.
(2021). Hunters, busybodies and the knowledge network building associated
with deprivation curiosity. *Nature Human Behaviour, 5*, 327–336.
https://doi.org/10.1038/s41562-020-00985-7

Mitchell, A., Jurkowitz, M., Oliphant, J. B., & Shearer, E. (2024, April 14). Most Americans have heard of the conspiracy theory that the COVID-19 outbreak was planned, and about one-third of those aware of it say it might be true. Pew Research Center. https://www.pewresearch.org/journalism/2020/06/29/most-americans-have-heard-of-the-conspiracy-theory-that-the-covid-19-outbreak-was-planned-and-about-one-third-of-those-aware-of-it-say-it-might-be-true/

Shulman, H. C., & Sweitzer, M. D. (2018). Varying metacognition through public opinion questions: How language can affect political engagement. Journal of Language and Social Psychology, 37(2), 224-237.

Shulman, H. C., & Sweitzer, M. D. (2018). Advancing framing theory: Designing an equivalency frame to improve political information processing. Human Communication Research, 44(2), 155-175.

Wells, A., & Cartwright-Hatton, S. (2004). A short form of the
metacognitions questionnaire: Properties of the MCQ-30.
*Behaviour Research and Therapy, 42*(4), 385–396.

Xue, H., Oh, Y. J., Zhou, X., Zhang, X., & Oxley, B. (2026). Users'
prompting strategies and ChatGPT's contextual adaptation shape
conversational information-seeking experiences. *Scientific Reports, 16*,
12112. https://doi.org/10.1038/s41598-026-42465-4

---

## Credits

Adapted from [CeciliaZhu1997/Chatbot-Experiment](https://github.com/CeciliaZhu1997/Chatbot-Experiment).

---

## License

MIT License — see [LICENSE](LICENSE) for details.
