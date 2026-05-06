# AI Metacognition Project

A longitudinal 21-day experiment platform that embeds an OpenAI LLM conversation inside a Qualtrics survey. This repository documents the pipeline for a single-day session — repeated across 21 days.

---

## Pipeline

```
Qualtrics Pre-survey
  ↓  pid + topic + session_num passed via embedded data
Embedded LLM Chat (JavaScript inside Qualtrics block)
  ↓  calls https://metacog-ai.onrender.com/chat
Flask App (pipeline.py) — OpenAI API
  ↓  logs transcript to SQLite
Post-survey measures continue in same Qualtrics session
```

---

## File Structure

```
metacog_ai/
├── pipeline.py
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
| Topic conditions | `conspiracies`, `info_seeking` |
| Study design | Longitudinal, 21 days |

---

## Topic Conditions

Participants are randomly assigned to one of two topic conditions at the start of Day 1 and remain in that condition for all 21 days:

| Condition | Topic |
|---|---|
| `political_conspiracies` | controversial |
| `selective_exposure` | non-controversial |

---

## Qualtrics Survey Flow (Single-Day Example)

This is the structure for one daily session. The same survey is distributed on each of the 21 days, with `session_num` updated each wave.

```
Set Embedded Data
  session_num = 1  ← increment each wave (1–21)
  topic = (empty, set by randomizer)
↓
Randomizer (50/50, evenly distributed)
  ├── topic = political_conspiracies
  └── topic = selective_exposure
↓
ID/Student No.
Consent form
↓
Branch: IF session_num = 1
  └── Demographic block
Branch: IF topic = conspiracies
  └── Topic_Political_Conspiracy block
Branch: IF topic = info_seeking
  └── Topic_Selective_Exposure block
↓
Metacognition Questionnaire (MCQ-30; Wells & Cartwright-Hatton, 2004)
↓
LLM_Chatbot block — Session  (5 exchanges)
↓
Metacognitive Experience_1 (Shulman & Sweitzer, 2018)
↓
Confidence_Scale_1
↓
End of Survey
```
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

> **Note:** Render's free tier spins down after inactivity. The first request after a period of inactivity may take 30–50 seconds. Consider upgrading for live data collection.

---

## Data Export

Download all LLM conversation logs as CSV:

```
https://metacog-ai.onrender.com/export?token=your-export-token
```

Merge with Qualtrics exports using `pid` + `session_num` as the join key.

---

## Dependencies

```
openai
flask
flask-cors
gunicorn
```

---

## Credits

Adapted from [CeciliaZhu1997/Chatbot-Experiment](https://github.com/CeciliaZhu1997/Chatbot-Experiment).

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
