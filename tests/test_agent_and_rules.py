import pytest

from csva.agent import ClaimStatusAgent, redact
from csva.benchmark import score
from csva.brains import RuleBrain, fmt_dob, heard_since_last_agent_turn
from csva.mock_insurer import MockInsurerCall, load_db
from csva.models import Action, ClaimQuery, Turn

from .conftest import SCEN, run_scenario


@pytest.mark.parametrize("sid", list(SCEN))
def test_every_scenario_succeeds_and_matches_expected(sid):
    _, rec = run_scenario(sid)
    assert rec.success, rec.transcript[-3:]
    ok, tot = score(rec, SCEN[sid]["expect"])
    assert ok == tot, (rec.result, SCEN[sid]["expect"])


def test_flaky_input_is_retried_once():
    _, rec = run_scenario("ivr_flaky_input_retry")
    said = [t.text for t in rec.transcript if t.speaker == "agent"]
    assert said.count("[DTMF 123456789#]") == 2


def test_agent_stays_silent_through_hold_music():
    _, rec = run_scenario("ivr_paid")
    assert sum(1 for t in rec.transcript if t.speaker == "agent" and t.text == "[silence]") >= 2


def test_rep_path_presses_zero_then_talks():
    _, rec = run_scenario("rep_denied")
    said = [t.text for t in rec.transcript if t.speaker == "agent"]
    assert said[0] == "[DTMF 0]"
    assert any("NPI is 1234567890" in s for s in said)
    assert said[-1].startswith("No, that's all")


def test_agent_never_discloses_another_patients_identifiers():
    _, rec = run_scenario("rep_paid")
    spoken = " ".join(t.text for t in rec.transcript if t.speaker == "agent")
    assert "987654321" not in spoken and "11302001" not in spoken


def test_agent_gives_up_instead_of_looping_forever():
    class Mute:
        name = "mute"

        def decide(self, q, tr):
            return Action(type="wait")

    q = ClaimQuery(**SCEN["ivr_paid"]["query"])
    rec = ClaimStatusAgent(Mute(), max_turns=10).run(q, MockInsurerCall(load_db()))
    assert rec.turns == 10 and not rec.success and rec.result is None


def test_on_turn_callback_sees_every_turn_in_order():
    seen: list[Turn] = []
    _, rec = run_scenario("ivr_paid", on_turn=seen.append)
    assert [t.text for t in seen] == [t.text for t in rec.transcript]
    assert [t.t_ms for t in seen] == sorted(t.t_ms for t in seen)


def test_record_metadata():
    _, rec = run_scenario("ivr_pending")
    assert rec.brain == "rules" and rec.scenario_id == "ivr_pending"
    assert len(rec.decision_ms) == rec.turns > 0


def test_success_requires_the_call_to_end_cleanly():
    class NeverHangsUp:
        """Navigates normally, gets the answer, then refuses to end the call."""

        name = "x"

        def __init__(self):
            self.inner = RuleBrain()

        def decide(self, q, tr):
            a = self.inner.decide(q, tr)
            return Action(type="wait", result=a.result) if a.result is not None else a

    q = ClaimQuery(**SCEN["ivr_paid"]["query"])
    rec = ClaimStatusAgent(NeverHangsUp(), max_turns=25).run(q, MockInsurerCall(load_db()))
    assert rec.result is not None and not rec.success


class TestRedaction:
    q = ClaimQuery(**SCEN["rep_denied"]["query"])

    def test_masks_member_id_dob_and_name(self):
        out = redact("Jane DOE member 123456789 dob 04/12/1985 or 04121985", self.q)
        assert "123456789" not in out and "1985" not in out and "doe" not in out.lower()
        assert "[MEMBER_ID]" in out and "[DOB]" in out and "[PATIENT]" in out

    def test_keeps_non_phi_text(self):
        assert redact("Status: denied. REF-123", self.q) == "Status: denied. REF-123"


class TestRuleBrainUnits:
    def test_heard_since_last_agent_turn(self):
        tr = [
            Turn(speaker="insurer", text="a"),
            Turn(speaker="agent", text="x"),
            Turn(speaker="insurer", text="b"),
            Turn(speaker="insurer", text="c"),
        ]
        assert heard_since_last_agent_turn(tr) == "b c"

    def test_fmt_dob(self):
        assert fmt_dob("04121985") == "04/12/1985"

    def test_retry_is_bounded(self):
        q = ClaimQuery(**SCEN["ivr_paid"]["query"])
        b = RuleBrain()
        tr = [Turn(speaker="insurer", text="Please enter the member ID, numbers only, followed by the pound key.")]
        first = b.decide(q, tr)
        assert first.type == "dtmf"
        tr2 = [Turn(speaker="insurer", text="Sorry, I didn't get that.")]
        kinds = [b.decide(q, tr2).type for _ in range(6)]
        assert kinds[:3] == ["dtmf"] * 3 and kinds[3:] == ["wait"] * 3

    def test_result_is_attached_to_every_action_after_it_is_heard(self):
        q = ClaimQuery(**SCEN["ivr_paid"]["query"])
        b = RuleBrain()
        tr = [Turn(speaker="insurer", text="Status: paid. Paid amount $142.50. To end this call, press 9.")]
        a = b.decide(q, tr)
        assert a.digits == "9" and a.result is not None and a.result.status == "paid"
