"""Opt-in smoke tests against the REAL Gemini API (free tier).

Skipped by default and in normal CI. Run locally / via the manual CI job:
    GEMINI_API_KEY=... pytest -m live -o addopts=''
"""

import os

import pytest

from csva.agent import ClaimStatusAgent
from csva.brains import GeminiBrain, gemini_api_key
from csva.diagnostics import doctor

from .conftest import make_call

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not gemini_api_key(), reason="GEMINI_API_KEY not set"),
]


def test_doctor_is_green():
    assert all(ok for ok, _ in doctor())


@pytest.mark.parametrize("sid", ["ivr_denied_medical_necessity", "rep_paid"])
def test_real_model_completes_a_call(sid):
    q, call = make_call(sid)
    brain = GeminiBrain(rpm=float(os.getenv("CSVA_RPM", "8")))
    rec = ClaimStatusAgent(brain).run(q, call, sid)
    assert rec.success, rec.transcript[-4:]
    assert rec.result is not None and rec.result.status in {"denied", "paid"}
    assert brain.fallbacks_used <= 2  # a couple of hiccups tolerated; wholesale fallback is a failure
