# =============================================================================
# AI Metacognition Project — pipeline.py
# =============================================================================
# Flask backend for the longitudinal LLM conversation pipeline.
# Receives participants from Qualtrics pre-survey, serves a topic-locked
# OpenAI LLM session, logs transcripts to PostgreSQL, and redirects back to
# Qualtrics post-survey.
#
# Adapted from: CeciliaZhu1997/Chatbot-Experiment
#   github.com/CeciliaZhu1997/Chatbot-Experiment
# Study design: Lydon-Staley et al. (2021), Nature Human Behaviour
#
# Modified: SQLite replaced with PostgreSQL for persistent storage on Render
# =============================================================================

import os
import csv
import io
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_file, redirect, abort
from flask_cors import CORS
import psycopg2
from openai import OpenAI

app = Flask(__name__)
CORS(app)  # Allow Qualtrics (cross-origin) to call /chat

# ── Config ───────────────────────────────────────────────────────────────────

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

MODEL     = "gpt-4o-mini"
MAX_TURNS = 5
RETURN_URL = "https://ucdavis.co1.qualtrics.com/jfe/form/SV_71mW9DfPrudBg0K"
EXPORT_TOKEN = os.environ.get("EXPORT_TOKEN", "change-me-before-launch")

# ── System prompts ────────────────────────────────────────────────────────────

SYSTEM_PROMPTS = {
    "conspiracies": (
        "You are a knowledgeable, neutral, and engaging conversation partner "
        "for a research study on how people think about political conspiracy theories. "
        "The participant has been asked to seek information about the conspiracy "
        "theory that COVID-19 was intentionally planned by powerful people."
        "\n\n"
        "Your role:\n"
        "- Help the participant explore what they know, what they are uncertain "
        "about, and what evidence exists around this conspiracy theory.\n"
        "- Discuss related topics such as misinformation, epistemic trust, and "
        "how people evaluate information sources. Stay within this domain.\n"
        "- Do NOT tell the participant what to believe. Facilitate reflection, "
        "not persuasion.\n"
        "- When the participant states a belief, ask them to reflect on where "
        "that belief comes from and what they find convincing.\n"
        "- When the participant has a knowledge gap, encourage them to think "
        "about what they already know and what they want to understand better.\n"
        "- End each response with one focused follow-up question.\n"
        "- Keep responses to 3 to 5 sentences. Plain, conversational language."
    ),
    "info_seeking": (
        "You are a knowledgeable, neutral, and engaging conversation partner "
        "for a research study on the psychology of information seeking. "
        "The participant has been asked to learn about why people seek information "
        "differently, how curiosity works, and how people build knowledge over time."
        "\n\n"
        "Your role:\n"
        "- Help the participant explore concepts related to curiosity, information "
        "seeking, knowledge networks, and learning behavior. Stay within this domain.\n"
        "- Encourage the participant to reflect on their own information-seeking "
        "habits — how they look things up, what drives their curiosity, and how "
        "they tend to explore topics.\n"
        "- When the participant has a knowledge gap, encourage them to think "
        "about what they already know and what they want to understand better.\n"
        "- End each response with one focused follow-up question.\n"
        "- Keep responses to 3 to 5 sentences. Plain, conversational language."
    ),
}

VALID_TOPICS = set(SYSTEM_PROMPTS.keys())

# ── Database — PostgreSQL ─────────────────────────────────────────────────────
# Reads DATABASE_URL from environment variable set on Render.
# Render provides 'postgres://' — psycopg2 requires 'postgresql://'

def get_conn():
    """Open a new PostgreSQL connection."""
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise RuntimeError("DATABASE_URL is not set")
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    try:
        return psycopg2.connect(url)
    except Exception as e:
        raise RuntimeError(f"Database connection failed: {e}")


def init_db():
    """Create tables if they do not exist. Called once on startup."""
    con = get_conn()
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id          SERIAL PRIMARY KEY,
            session_id  TEXT    NOT NULL,
            pid         TEXT,
            topic       TEXT,
            session_num INTEGER,
            role        TEXT    NOT NULL,
            content     TEXT    NOT NULL,
            turn        INTEGER,
            timestamp   TEXT    NOT NULL
        )
    """)
    con.commit()
    cur.close()
    con.close()


def save_message(session_id, pid, topic, session_num, role, content, turn):
    con = get_conn()
    cur = con.cursor()
    cur.execute(
        """INSERT INTO messages
           (session_id, pid, topic, session_num, role, content, turn, timestamp)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
        (session_id, pid, topic, session_num, role, content, turn,
         datetime.utcnow().isoformat())
    )
    con.commit()
    cur.close()
    con.close()


def get_history(session_id):
    con = get_conn()
    cur = con.cursor()
    cur.execute(
        "SELECT role, content FROM messages WHERE session_id=%s ORDER BY id",
        (session_id,)
    )
    rows = cur.fetchall()
    cur.close()
    con.close()
    return [{"role": r, "content": c} for r, c in rows]


def count_turns(session_id):
    con = get_conn()
    cur = con.cursor()
    cur.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id=%s AND role='user'",
        (session_id,)
    )
    n = cur.fetchone()[0]
    cur.close()
    con.close()
    return n


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    pid         = request.args.get("pid", "unknown")
    topic       = request.args.get("topic", "").strip().lower()
    session_num = request.args.get("session", "1")

    if topic not in VALID_TOPICS:
        abort(400, f"Invalid topic '{topic}'. Must be one of: {', '.join(VALID_TOPICS)}.")

    try:
        session_num = int(session_num)
    except ValueError:
        abort(400, "Session number must be an integer.")

    session_id = f"{pid}_s{session_num}"

    return render_template(
        "chat.html",
        pid=pid,
        topic=topic,
        session_num=session_num,
        session_id=session_id,
        max_turns=MAX_TURNS,
    )


@app.route("/chat", methods=["POST"])
def chat():
    data        = request.json
    user_msg    = data.get("message", "").strip()
    session_id  = data.get("session_id")
    pid         = data.get("pid", "unknown")
    topic       = data.get("topic", "").strip().lower()
    session_num = int(data.get("session_num", 1))

    if not user_msg or not session_id:
        return jsonify({"error": "missing fields"}), 400
    if topic not in VALID_TOPICS:
        return jsonify({"error": "invalid topic"}), 400

    turns = count_turns(session_id)
    if turns >= MAX_TURNS:
        return jsonify({"reply": None, "done": True, "turns_left": 0})

    save_message(session_id, pid, topic, session_num, "user", user_msg, turns + 1)

    history  = get_history(session_id)
    messages = [{"role": "system", "content": SYSTEM_PROMPTS[topic]}] + history

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=500,
        temperature=0.7,
    )
    reply = response.choices[0].message.content.strip()

    save_message(session_id, pid, topic, session_num, "assistant", reply, turns + 1)

    done = (turns + 1) >= MAX_TURNS
    return jsonify({
        "reply":      reply,
        "done":       done,
        "turns_left": max(0, MAX_TURNS - turns - 1),
    })


@app.route("/export")
def export():
    if request.args.get("token", "") != EXPORT_TOKEN:
        abort(403, "Invalid or missing export token.")

    con = get_conn()
    cur = con.cursor()
    cur.execute(
        """SELECT session_id, pid, topic, session_num,
                  role, content, turn, timestamp
           FROM messages ORDER BY session_id, id"""
    )
    rows = cur.fetchall()
    cur.close()
    con.close()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["session_id", "pid", "topic", "session_num",
                     "role", "content", "turn", "timestamp"])
    writer.writerows(rows)
    output.seek(0)

    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype="text/csv",
        as_attachment=True,
        download_name="metacog_sessions_export.csv"
    )
    
@app.route("/ping")
def ping():
    return "pong", 200

# ── Init & run ────────────────────────────────────────────────────────────────

init_db()

if __name__ == "__main__":
    app.run(debug=True)
