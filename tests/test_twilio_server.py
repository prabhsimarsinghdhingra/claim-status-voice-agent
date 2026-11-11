import re
from urllib.parse import urlencode

import pytest
from fastapi.testclient import TestClient

from csva import twilio_server
from csva.db import connect
from csva.mock_insurer import MockInsurerCall, load_db

from .conftest import SCEN


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(twilio_server, "DB_PATH", str(tmp_path / "calls.db"))
    twilio_server.SESSIONS.clear()
    return TestClient(twilio_server.app), tmp_path / "calls.db"


def post(c, path, **data):
    return c.post(path, content=urlencode(data), headers={"content-type": "application/x-www-form-urlencoded"})


def twiml_to_action(xml):
    if m := re.search(r'<Play digits="([^"]*)"', xml):
        return {"type": "dtmf", "digits": m.group(1).replace("w", "")}
    if m := re.search(r"<Say[^>]*>(.*?)</Say>", xml):
        return {"type": "say", "text": m.group(1)}
    return {"type": "wait"}


def drive_call(c, case, sid="CA1"):
    insurer = MockInsurerCall(load_db(), flaky=SCEN[case]["flaky"])
    assert "<Gather" in post(c, f"/twiml/start?case={case}", CallSid=sid).text
    heard = " ".join(insurer.open())
    xml = ""
    for _ in range(60):
        xml = post(c, "/twiml/turn", CallSid=sid, SpeechResult=heard).text
        if "<Hangup/>" in xml:
            return xml
        heard = " ".join(insurer.step(twiml_to_action(xml)))
    return xml


def test_health(client):
    c, _ = client
    assert c.get("/health").json() == {"ok": True, "active_calls": 0}


@pytest.mark.parametrize("case", ["ivr_denied_medical_necessity", "ivr_paid", "rep_denied", "ivr_flaky_input_retry"])
def test_full_call_through_webhooks_is_persisted(client, case):
    c, db = client
    assert "<Hangup/>" in drive_call(c, case)
    row = connect(db).execute("select status, success from calls").fetchone()
    assert row == (SCEN[case]["expect"]["status"], 1)
    assert twilio_server.SESSIONS == {}


def test_unknown_session_hangs_up(client):
    c, _ = client
    assert "<Hangup/>" in post(c, "/twiml/turn", CallSid="ghost").text


def test_empty_speech_results_in_a_pause_not_a_crash(client):
    c, _ = client
    post(c, "/twiml/start", CallSid="CA2")
    xml = post(c, "/twiml/turn", CallSid="CA2", SpeechResult="").text
    assert "<Pause" in xml and "<Gather" in xml


def test_status_callback_persists_an_interrupted_call(client):
    c, db = client
    post(c, "/twiml/start", CallSid="CA3")
    post(c, "/twiml/turn", CallSid="CA3", SpeechResult="Thank you for calling. For claim status, press 2.")
    assert post(c, "/twiml/status", CallSid="CA3", CallStatus="completed").json() == {"ok": True}
    assert connect(db).execute("select success from calls").fetchone() == (0,)
    assert "CA3" not in twilio_server.SESSIONS


def test_twiml_escapes_xml_in_spoken_text():
    assert twilio_server.escape("a & b <c>") == "a &amp; b &lt;c&gt;"
