from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .agent import redact
from .models import CallRecord, ClaimQuery

SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  scenario_id TEXT, brain TEXT, claim_number TEXT,
  success INTEGER, status TEXT, denial_code TEXT, paid_amount REAL,
  reference_number TEXT, next_action TEXT,
  turns INTEGER, duration_ms INTEGER, mean_decision_ms REAL,
  result_json TEXT, transcript_json TEXT
);
"""


def connect(path: str | Path = "calls.db") -> sqlite3.Connection:
    con = sqlite3.connect(str(path))
    con.executescript(SCHEMA)
    return con


def save_call(con: sqlite3.Connection, rec: CallRecord, q: ClaimQuery) -> int:
    r = rec.result
    mean_ms = sum(rec.decision_ms) / len(rec.decision_ms) if rec.decision_ms else 0.0
    transcript = [{**t.model_dump(), "text": redact(t.text, q)} for t in rec.transcript]
    cur = con.execute(
        """INSERT INTO calls (scenario_id, brain, claim_number, success, status, denial_code, paid_amount,
           reference_number, next_action, turns, duration_ms, mean_decision_ms, result_json, transcript_json)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            rec.scenario_id,
            rec.brain,
            q.claim_number,
            int(rec.success),
            r.status if r else None,
            r.denial_code if r else None,
            r.paid_amount if r else None,
            r.reference_number if r else None,
            r.next_action if r else None,
            rec.turns,
            rec.duration_ms,
            mean_ms,
            r.model_dump_json() if r else None,
            json.dumps(transcript),
        ),
    )
    con.commit()
    return int(cur.lastrowid or 0)
