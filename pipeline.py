# =============================================================================
# AI Metacognition Project — pipeline.py
# =============================================================================
# Flask backend for the longitudinal LLM conversation pipeline.
# Receives participants from Qualtrics pre-survey, serves a topic-locked
# OpenAI LLM session, logs transcripts to SQLite, and redirects back to
# Qualtrics post-survey.
#
# Adapted from: CeciliaZhu1997/Chatbot-Experiment
#   github.com/CeciliaZhu1997/Chatbot-Experiment
# Study design: Lydon-Staley et al. (2021), Nature Human Behaviour
# =============================================================================

import os
import sqlite3
import csv
import io
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_file, redirect, abort

from openai import OpenAI

app = Flask(__name__)

# ── Config ───────────────────────────────────────────────────────────────────

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

MODEL     = "gpt-4o-mini"           # OpenAI model
MAX_TURNS = 15                       # Max user turns per session (~10–15 min)
DB_PATH   = "metacog_sessions.db"   # SQLite database file

# Qualtrics post-survey URL — pid and session_num appended on redirect
RETURN_URL = "https://ucdavis.co1.qualtrics.com/jfe/form/SV_71mW9DfPrudBg0K"

# Protects the /export endpoint — set via environment variable on Render
EXPORT_TOKEN = os.environ.get("EXPORT_TOKEN", "change-me-before-launch")

# ── System prompts ────────────────────────────────────────────────────────────
# Topic-locked per Lydon-Staley et al. (2021) design.
# The LLM stays within the assigned domain across all turns, encouraging
# iterative, gap-filling conversation that builds a tight knowledge network.

SYSTEM_PROMPTS = {
    "conspiracies": (
        "You are a knowledgeable and engaging conversation partner. "
        "Discuss only topics related to political conspiracies, misinformation, "
        "epistemic trust, media literacy, and related phenomena. "
        "Do not answer questions outside this domain — if the participant asks "
        "about something unrelated, gently redirect them back to the topic. "
        "When the participant seems uncertain, encourage them to reflect on what "
        "they already know and what they would like to understand better. "
        "Ask follow-up questions to deepen their engagement with the topic."
    ),
    "info_seeking": (
        "You are a knowledgeable and engaging conversation partner. "
        "Discuss only topics related to the psychology of information seeking, "
        "curiosity, knowledge networks, learning behavior, and related phenomena. "
        "Do not answer questions outside this domain — if the participant asks "
        "about something unrelated, gently redirect them back to the topic. "
        "When the participant seems uncertain, encourage them to reflect on what "
        "they already know and what they would like to understand better. "
        "Ask follow-up questions to deepen their engagement with the topic."
    ),
}

VALID_TOPICS = set(SYSTEM_PROMPTS.keys())

# ── Database ──────────────────────────────────────────────────────────────────

def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
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
    con.close()


def save_message(session_id, pid, topic, session_num, role, content, turn):
    con = sqlite3.connect(DB_PATH)
    con.execute(
        """INSERT INTO messages
           (session_id, pid, topic, session_num, role, content, turn, timestamp)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (session_id, pid, topic, session_num, role, content, turn,
         datetime.utcnow().isoformat())
    )
    con.commit()
    con.close()


def get_history(session_id):
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT role, content FROM messages WHERE session_id=? ORDER BY id",
        (session_id,)
    ).fetchall()
    con.close()
    return [{"role": r, "content": c} for r, c in rows]


def count_turns(session_id):
    con = sqlite3.connect(DB_PATH)
    n = con.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id=? AND role='user'",
        (session_id,)
    ).fetchone()[0]
    con.close()
    return n

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """
    Entry point — receives participants redirected from Qualtrics pre-survey.

    Expected URL parameters:
        pid         Qualtrics ResponseID
        topic       'conspiracies' or 'info_seeking'
        session     Session number (1, 2, 3 ...)
    """
    pid         = request.args.get("pid", "unknown")
    topic       = request.args.get("topic", "").strip().lower()
    session_num = request.args.get("session", "1")

    if topic not in VALID_TOPICS:
        abort(400, f"Invalid topic '{topic}'. Must be one of: {', '.join(VALID_TOPICS)}.")

    try:
        session_num = int(session_num)
    except ValueError:
        abort(400, "Session number must be an integer.")

    # Unique session ID links this participant + session across the database
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
    """
    Receives a user message, calls OpenAI, logs both turns, returns the reply.

    Expected JSON body:
        pid, topic, session_num, session_id, message
    """
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

    # Log user message
    save_message(session_id, pid, topic, session_num, "user", user_msg, turns + 1)

    # Build full message history for OpenAI
    history  = get_history(session_id)
    messages = [{"role": "system", "content": SYSTEM_PROMPTS[topic]}] + history

    # Call OpenAI
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=500,
        temperature=0.7,
    )
    reply = response.choices[0].message.content.strip()

    # Log assistant reply
    save_message(session_id, pid, topic, session_num, "assistant", reply, turns + 1)

    done = (turns + 1) >= MAX_TURNS
    return jsonify({
        "reply":      reply,
        "done":       done,
        "turns_left": max(0, MAX_TURNS - turns - 1),
    })


@app.route("/export")
def export():
    """
    Download all conversation logs as CSV.
    Protected by EXPORT_TOKEN environment variable.
    Access: /export?token=your-secret-token
    """
    if request.args.get("token", "") != EXPORT_TOKEN:
        abort(403, "Invalid or missing export token.")

    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        """SELECT session_id, pid, topic, session_num,
                  role, content, turn, timestamp
           FROM messages ORDER BY session_id, id"""
    ).fetchall()
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


# ── Init & run ────────────────────────────────────────────────────────────────

init_db()

if __name__ == "__main__":
    app.run(debug=True)  # Set debug=False before deploying to production
