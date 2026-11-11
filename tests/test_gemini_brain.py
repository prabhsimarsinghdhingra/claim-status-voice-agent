import json

import pytest

from csva.agent import ClaimStatusAgent
from csva.brains import (
    DEFAULT_GEMINI_MODEL,
    SYSTEM_PROMPT,
    GeminiBrain,
    LLMConfigError,
    _extract_json,
    status_code_of,
)
from csva.mock_insurer import MockInsurerCall, load_db
from csva.models import ClaimQuery

from .conftest import SCEN, FakeGemini, HttpError, make_call, oracle, run_scenario


# ---------------------------------------------------------------- helpers
class TestExtractJson:
    def test_plain(self):
        assert _extract_json('{"type":"wait"}') == {"type": "wait"}

    def test_markdown_fence(self):
        assert _extract_json('```json\n{"type":"wait"}\n```') == {"type": "wait"}

    def test_prose_around_json(self):
        assert _extract_json('Sure! {"type":"dtmf","digits":"2"} done') == {"type": "dtmf", "digits": "2"}

    @pytest.mark.parametrize("bad", ["", "no json here", "{not json}"])
    def test_garbage_raises(self, bad):
        with pytest.raises(ValueError):
            _extract_json(bad)


class TestStatusCode:
    def test_attribute(self):
        assert status_code_of(HttpError(429)) == 429

    def test_from_message(self):
        assert status_code_of(RuntimeError("503 UNAVAILABLE: overloaded")) == 503

    def test_none(self):
        assert status_code_of(RuntimeError("weird")) is None


# ---------------------------------------------------------------- construction
class TestConstruction:
    def test_missing_key_is_a_config_error(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with pytest.raises(LLMConfigError, match="GEMINI_API_KEY"):
            GeminiBrain()

    def test_real_sdk_client_builds_offline_with_either_env_var(self, monkeypatch):
        pytest.importorskip("google.genai")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "dummy-key")
        assert GeminiBrain().model == DEFAULT_GEMINI_MODEL

    def test_model_comes_from_env(self, monkeypatch):
        monkeypatch.setenv("GEMINI_MODEL", "gemini-test-9")
        assert GeminiBrain(client=FakeGemini(lambda p, n: "")).model == "gemini-test-9"

    def test_explicit_model_beats_env(self, monkeypatch):
        monkeypatch.setenv("GEMINI_MODEL", "from-env")
        assert GeminiBrain(model="explicit", client=FakeGemini(lambda p, n: "")).model == "explicit"

    def test_generation_config_is_valid_for_the_real_sdk(self):
        types = pytest.importorskip("google.genai.types")
        cfg = types.GenerateContentConfig(**GeminiBrain.generation_config())
        assert cfg.response_mime_type == "application/json" and cfg.temperature == 0
        assert cfg.system_instruction == SYSTEM_PROMPT


# ---------------------------------------------------------------- happy path
@pytest.mark.parametrize("sid", list(SCEN))
def test_perfect_model_passes_every_scenario(sid, gemini_factory):
    q, call = make_call(sid)
    brain, client = gemini_factory(oracle(q))
    rec = ClaimStatusAgent(brain).run(q, call, sid)
    assert rec.success and brain.fallbacks_used == 0 and brain.grounding_drops == 0
    assert rec.brain == "gemini"
    assert brain.api_calls == len(client.calls) == rec.turns
    exp = SCEN[sid]["expect"]
    assert rec.result is not None
    for k, v in exp.items():
        assert getattr(rec.result, k) == v


def test_request_uses_json_mode_and_the_case_file(gemini_factory):
    q, call = make_call("ivr_paid")
    brain, client = gemini_factory(oracle(q))
    ClaimStatusAgent(brain).run(q, call)
    first = client.calls[0]
    assert first["model"] == "test-model"
    assert first["config"]["response_mime_type"] == "application/json"
    assert "CASE FILE" in first["contents"] and q.claim_number in first["contents"]


def test_model_output_with_markdown_fences_is_accepted(gemini_factory):
    q, call = make_call("ivr_paid")
    inner = oracle(q)
    brain, _ = gemini_factory(lambda p, n: "```json\n" + inner(p, n) + "\n```")
    assert ClaimStatusAgent(brain).run(q, call).success and brain.fallbacks_used == 0


# ---------------------------------------------------------------- degradation
def test_garbage_output_falls_back_but_is_counted(gemini_factory):
    q, call = make_call("ivr_denied_medical_necessity")
    brain, _ = gemini_factory(lambda p, n: "I'm sorry, I can't help with that.")
    rec = ClaimStatusAgent(brain).run(q, call)
    assert rec.success and brain.fallbacks_used == rec.turns


def test_schema_invalid_json_falls_back(gemini_factory):
    q, call = make_call("ivr_paid")
    brain, _ = gemini_factory(lambda p, n: json.dumps({"type": "teleport"}))
    assert ClaimStatusAgent(brain).run(q, call).success and brain.fallbacks_used > 0


def test_empty_or_blocked_response_falls_back(gemini_factory):
    q, call = make_call("ivr_paid")
    brain, _ = gemini_factory(lambda p, n: None)
    assert ClaimStatusAgent(brain).run(q, call).success and brain.fallbacks_used > 0


def test_transient_429_is_retried_with_exponential_backoff(gemini_factory, no_sleep):
    slept, _ = no_sleep
    q, call = make_call("ivr_paid")
    good = oracle(q)
    state = {"fails": 0}

    def flaky(prompt, n):
        if state["fails"] < 2:
            state["fails"] += 1
            raise HttpError(429, "RESOURCE_EXHAUSTED")
        return good(prompt, n)

    brain, _ = gemini_factory(flaky)
    rec = ClaimStatusAgent(brain).run(q, call)
    assert rec.success and brain.retries == 2 and brain.fallbacks_used == 0
    assert slept[:2] == [2.0, 4.0]


def test_retries_exhausted_falls_back(gemini_factory):
    q, call = make_call("ivr_paid")

    def always_503(prompt, n):
        raise HttpError(503, "UNAVAILABLE")

    brain, _ = gemini_factory(always_503, max_retries=2)
    rec = ClaimStatusAgent(brain).run(q, call)
    assert rec.success and brain.fallbacks_used == rec.turns and brain.retries == 2 * rec.turns


@pytest.mark.parametrize("code", [400, 401, 403, 404])
def test_fatal_errors_abort_loudly_instead_of_silently_degrading(gemini_factory, code):
    q, call = make_call("ivr_paid")

    def reject(prompt, n):
        raise HttpError(code, "API key not valid / model not found")

    brain, client = gemini_factory(reject)
    with pytest.raises(LLMConfigError, match=str(code)):
        ClaimStatusAgent(brain).run(q, call)
    assert len(client.calls) == 1  # did not hammer the API


def test_unknown_exception_falls_back(gemini_factory):
    q, call = make_call("ivr_paid")

    def boom(prompt, n):
        raise ConnectionError("dns failure")

    brain, _ = gemini_factory(boom)
    assert ClaimStatusAgent(brain).run(q, call).success and brain.fallbacks_used > 0


# ---------------------------------------------------------------- rate limiting
def test_rpm_throttle_spaces_requests(no_sleep):
    slept, _ = no_sleep
    now = {"t": 100.0}

    def clock():
        return now["t"]

    def fake_sleep(s):
        slept.append(s)
        now["t"] += s

    q = ClaimQuery(**SCEN["ivr_paid"]["query"])
    brain = GeminiBrain(model="m", client=FakeGemini(oracle(q)), rpm=30, sleep=fake_sleep, clock=clock)
    ClaimStatusAgent(brain).run(q, MockInsurerCall(load_db()))
    assert brain.api_calls > 3
    assert all(abs(s - 2.0) < 1e-6 for s in slept) and len(slept) == brain.api_calls - 1


def test_no_rpm_means_no_throttle_sleep(gemini_factory, no_sleep):
    slept, _ = no_sleep
    q, call = make_call("ivr_paid")
    brain, _ = gemini_factory(oracle(q))
    ClaimStatusAgent(brain).run(q, call)
    assert slept == []


# ---------------------------------------------------------------- hallucination defences
def test_result_claimed_before_payer_said_anything_is_discarded(gemini_factory):
    q, call = make_call("ivr_paid")
    fake = {"status": "paid", "paid_amount": 1.0, "reference_number": "REF-99999"}
    good = oracle(q)

    def early_liar(prompt, n):
        d = json.loads(good(prompt, n))
        if n == 0:
            d["result"] = fake
        return json.dumps(d)

    brain, _ = gemini_factory(early_liar)
    rec = ClaimStatusAgent(brain).run(q, call)
    assert rec.result is not None and rec.result.reference_number != "REF-99999" and rec.result.paid_amount == 142.5


def test_wrong_model_values_are_overridden_by_what_the_payer_said(gemini_factory):
    q, call = make_call("ivr_denied_medical_necessity")

    def corrupt(d):
        if d.get("result"):
            d["result"].update(status="paid", paid_amount=999.99, denial_code="CO-97", reference_number="REF-00000")

    brain, _ = gemini_factory(oracle(q, mutate=corrupt))
    rec = ClaimStatusAgent(brain).run(q, call)
    r = rec.result
    assert r is not None
    assert (r.status, r.denial_code, r.paid_amount) == ("denied", "CO-50", 0.0)
    assert r.reference_number and r.reference_number != "REF-00000"


def test_model_invented_check_number_is_dropped_by_grounding(gemini_factory):
    q, call = make_call("ivr_denied_medical_necessity")

    def invent(d):
        if d.get("result"):
            d["result"]["check_number"] = "EFT7777777"

    brain, _ = gemini_factory(oracle(q, mutate=invent))
    rec = ClaimStatusAgent(brain).run(q, call)
    assert rec.result is not None and rec.result.check_number is None
    assert brain.grounding_drops > 0


def test_works_when_model_omits_result_and_only_picks_actions(gemini_factory):
    q, call = make_call("ivr_pending")
    brain, _ = gemini_factory(oracle(q, include_result=False))
    rec = ClaimStatusAgent(brain).run(q, call)
    assert rec.success and rec.result is not None and rec.result.status == "pending"


def test_rule_baseline_and_perfect_model_agree():
    """Sanity: our fake 'perfect model' is equivalent to the rule brain on the happy path."""
    for sid in SCEN:
        _, rb = run_scenario(sid)
        q, call = make_call(sid)
        brain = GeminiBrain(model="m", client=FakeGemini(oracle(q)))
        gb = ClaimStatusAgent(brain).run(q, call, sid)
        assert rb.result == gb.result


def test_latency_metric_excludes_throttle_and_backoff_waits():
    """Real-time test: with a 0.05s/request throttle, reported decision latency must still be ~0, not ~50ms."""
    import time

    q = ClaimQuery(**SCEN["ivr_paid"]["query"])
    brain = GeminiBrain(model="m", client=FakeGemini(oracle(q)), rpm=1200, sleep=time.sleep)
    t0 = time.perf_counter()
    rec = ClaimStatusAgent(brain).run(q, MockInsurerCall(load_db()))
    wall = time.perf_counter() - t0
    assert wall > 0.25, "throttle should have really slept"
    assert max(rec.decision_ms) < 30, rec.decision_ms
    assert sum(rec.decision_ms) / 1000 < wall / 3


def test_backoff_wait_is_tracked_per_decision(gemini_factory):
    from csva.models import Turn

    q = ClaimQuery(**SCEN["ivr_paid"]["query"])
    good = oracle(q)
    state = {"n": 0}

    def once_429(prompt, n):
        state["n"] += 1
        if state["n"] == 1:
            raise HttpError(429)
        return good(prompt, n)

    brain, _ = gemini_factory(once_429)
    heard = [Turn(speaker="insurer", text="For claim status, press 2.")]
    brain.decide(q, heard)
    assert brain.last_wait_ms == 2000.0  # first backoff step
    brain.decide(q, heard)
    assert brain.last_wait_ms == 0.0  # reset for the next decision
