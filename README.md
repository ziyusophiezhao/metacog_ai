# AI Metacognition Project

A longitudinal experiment platform that connects a Qualtrics survey to an OpenAI LLM session and back to Qualtrics for post-session measures.

---

## Pipeline

```
Qualtrics Pre-survey
  ↓  redirect with pid + topic + session_num
Flask App (pipeline.py) — OpenAI LLM session
  ↓  redirect with pid + session_num
Qualtrics Post-survey
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

## Setup

```bash
git clone https://github.com/your-lab/metacog_ai.git
cd metacog_ai
pip install -r requirements.txt
export OPENAI_API_KEY="sk-your-key-here"
python pipeline.py
```

---

## Qualtrics Implementation

### Step 1 — Pre-survey: redirect to the LLM app

At the end of your pre-survey, set the **End of Survey** element to redirect to:

```
https://your-app-url.onrender.com/?pid=${e://Field/ResponseID}&topic=${e://Field/topic}&session=${e://Field/session_num}
```

In **Survey Flow**, add these Embedded Data fields before your first question block:

| Field | Value |
|---|---|
| `topic` | Set by randomizer (e.g. `conspiracies` or `selective_exposure`) |
| `session_num` | Set manually per wave (e.g. `1`, `2`, `3`) |

**Survey Flow order:**

```
Set Embedded Data: session_num = 1
Set Embedded Data: topic (empty)
↓
Randomizer (50/50)
  → Set Embedded Data: topic = conspiracies
  → Set Embedded Data: topic = selective_exposure
↓
Block: Pre-survey questions
↓
End of Survey → Redirect to app URL above
```

---

### Step 2 — LLM session (pipeline.py)

The app receives `pid`, `topic`, and `session_num` from the URL, runs the LLM conversation for up to 15 turns, then automatically redirects the participant to the post-survey.

No Qualtrics configuration needed here — this is handled by `pipeline.py`.

---

### Step 3 — Post-survey: capture returning participant

In your post-survey **Survey Flow**, add these Embedded Data fields at the very top, set from URL parameters:

| Field | Source |
|---|---|
| `pid` | URL parameter `pid` |
| `session_num` | URL parameter `session_num` |
| `topic` | URL parameter `topic` |
| `complete` | URL parameter `complete` |

Qualtrics captures these automatically when the participant lands on the post-survey URL.

Use `pid` + `session_num` to merge post-survey responses with pre-survey responses and LLM conversation logs.

---

### Step 4 — Demographics (session 1 only)

In the post-survey, add a **Branch** at the end of the survey flow:

```
Branch: IF session_num = 1
  → Show Block: Demographics
```

This ensures demographics are collected once at baseline and skipped in later sessions.

---

## Deployment

1. Push repo to GitHub as `metacog_ai`
2. Create a Web Service on [Render.com](https://render.com)
3. Set environment variables: `OPENAI_API_KEY`, `EXPORT_TOKEN`
4. Start command: `gunicorn pipeline:app`

---

## Data Export

Download all LLM conversation logs at:

```
https://metacog-ai.onrender.com/export?token=your-export-token
```

Merge with Qualtrics exports using `pid` + `session_num` as the join key.

---

## Dependencies

```
openai
flask
gunicorn
```

---

## Credits

Adapted from [CeciliaZhu1997/Chatbot-Experiment](https://github.com/CeciliaZhu1997/Chatbot-Experiment).
Longitudinal design based on Lydon-Staley et al. (2021), *Nature Human Behaviour*.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
