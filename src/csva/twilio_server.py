"""Twilio webhook server: lets the same brain place a REAL phone call.

Flow (outbound):  scripts/place_call.py -> Twilio dials the number -> when answered Twilio POSTs /twiml/start
  -> we reply with <Gather input="speech dtmf"> (Twilio does the speech-to-text)
  -> Twilio POSTs what the other side said to /twiml/turn -> brain decides -> we reply with
     <Play digits=...> (keypad), <Say> (TTS), or <Pause>, then <Gather> again.
No audio streaming code needed; swap in Media Streams + Deepgram/ElevenLabs later for lower latency.

Only point this at numbers you own / a test IVR. Never at a real payer with real PHI unless you have the
BAA, authorization, and compliance sign-off.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from urllib.parse import parse_qs
from xml.sax.saxutils import escape

from fastapi import FastAPI, Request, Response

from .benchmark import load_scenarios
from .brains import Brain, GeminiBrain, RuleBrain
from .db import connect, save_call
from .models import CallRecord, ClaimQuery, ClaimStatusResult, Speaker, Turn

app = FastAPI(title="claim-status-voice-agent")
DB_PATH = os.getenv("CSVA_DB", "calls.db")


@dataclass
class Session:
    query: ClaimQuery
    brain: Brain
    scenario_id: str
    t0: float = field(default_factory=time.perf_counter)
    transcript: list = field(default_factory=list)
    decision_ms: list = field(default_factory=list)
    result: ClaimStatusResult | None = None


SESSIONS: dict[str, Session] = {}


def _make_brain() -> Brain:
    if os.getenv("CSVA_BRAIN", "rules") == "gemini":
        return GeminiBrain(rpm=float(os.getenv("CSVA_RPM", "0")) or None)
    return RuleBrain()


def _xml(inner: str) -> Response:
    return Response(f'<?xml version="1.0" encoding="UTF-8"?><Response>{inner}</Response>', media_type="application/xml")


GATHER = (
    '<Gather input="speech dtmf" action="/twiml/turn" method="POST" speechTimeout="auto" '
    'timeout="6" actionOnEmptyResult="true" speechModel="phone_call" enhanced="true"/>'
)


def _log(s: Session, speaker: Speaker, text: str) -> None:
    s.transcript.append(Turn(speaker=speaker, text=text, t_ms=int((time.perf_counter() - s.t0) * 1000)))


async def _form(request: Request) -> dict[str, str]:
    raw = (await request.body()).decode()
    return {k: v[0] for k, v in parse_qs(raw, keep_blank_values=True).items()}


@app.get("/health")
def health() -> dict:
    return {"ok": True, "active_calls": len(SESSIONS)}


@app.post("/twiml/start")
async def start(request: Request) -> Response:
    form = await _form(request)
    sid = form.get("CallSid", "local")
    case = request.query_params.get("case", "ivr_denied_medical_necessity")
    sc = next(s for s in load_scenarios() if s["id"] == case)
    SESSIONS[sid] = Session(ClaimQuery(**sc["query"]), _make_brain(), case)
    return _xml(GATHER)


@app.post("/twiml/turn")
async def turn(request: Request) -> Response:
    form = await _form(request)
    sid = form.get("CallSid", "local")
    s = SESSIONS.get(sid)
    if s is None:
        return _xml("<Hangup/>")

    heard = (form.get("SpeechResult") or "").strip()
    if heard:
        _log(s, "insurer", heard)

    t = time.perf_counter()
    action = s.brain.decide(s.query, s.transcript)
    s.decision_ms.append((time.perf_counter() - t) * 1000)
    _log(s, "agent", action.describe())
    if action.result:
        s.result = action.result

    if action.type == "dtmf":
        body = f'<Play digits="ww{escape(action.digits or "")}"/>'
    elif action.type == "say":
        body = f'<Say voice="Polly.Joanna">{escape(action.text or "")}</Say>'
    else:
        body = '<Pause length="2"/>'

    if s.result is not None and action.type != "wait":  # we have the answer and just sent the goodbye
        body += '<Pause length="1"/><Hangup/>'
        _finish(sid)
        return _xml(body)
    return _xml(body + GATHER)


@app.post("/twiml/status")
async def status(request: Request) -> dict:
    """Twilio statusCallback: persist if the remote side hung up before we finished."""
    form = await _form(request)
    _finish(form.get("CallSid", "local"))
    return {"ok": True}


def _finish(sid: str) -> None:
    s = SESSIONS.pop(sid, None)
    if s is None:
        return
    rec = CallRecord(
        scenario_id=s.scenario_id,
        brain=s.brain.name,
        success=s.result is not None,
        result=s.result,
        transcript=s.transcript,
        decision_ms=s.decision_ms,
        duration_ms=int((time.perf_counter() - s.t0) * 1000),
        turns=len(s.decision_ms),
    )
    save_call(connect(DB_PATH), rec, s.query)
