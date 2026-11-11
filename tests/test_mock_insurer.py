import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from csva.mock_insurer import HOLD, MENU, MockInsurerCall, call_ref, load_db


@pytest.fixture
def call():
    return MockInsurerCall(load_db())


def d(digits):
    return {"type": "dtmf", "digits": digits}


def say(text):
    return {"type": "say", "text": text}


WAIT = {"type": "wait"}


def walk_to_claim_prompt(c):
    c.step(d("2"))
    c.step(d("1234567890#"))
    c.step(d("123456789#"))
    return c.step(d("04121985"))


class TestIvr:
    def test_greeting_contains_menu(self, call):
        assert MENU in call.open()[0]

    def test_invalid_menu_choice_reprompts(self, call):
        assert call.step(d("7"))[0].startswith("Sorry")

    def test_unavailable_option(self, call):
        assert "not available" in call.step(d("1"))[0]

    def test_speech_can_select_claim_status(self, call):
        assert "N P I" in call.step(say("claim status please"))[0]

    def test_wrong_npi_rejected(self, call):
        call.step(d("2"))
        assert "didn't get that" in call.step(d("0000000000#"))[0]

    def test_unknown_member_rejected(self, call):
        call.step(d("2"))
        call.step(d("1234567890#"))
        assert "not valid" in call.step(d("555555555#"))[0]

    def test_wrong_dob_rejected(self, call):
        call.step(d("2"))
        call.step(d("1234567890#"))
        call.step(d("123456789#"))
        assert "could not verify" in call.step(d("01011990"))[0]

    def test_short_claim_number_rejected(self, call):
        walk_to_claim_prompt(call)
        assert "didn't get that" in call.step(d("123#"))[0]

    def test_hold_music_then_result(self, call):
        walk_to_claim_prompt(call)
        first = call.step(d("20260811002#"))
        assert HOLD in first
        assert call.step(WAIT) == [HOLD]
        out = call.step(WAIT)
        assert "Status: denied" in out[0] and call_ref("20260811002") in out[0]

    def test_keypad_during_hold_is_ignored(self, call):
        walk_to_claim_prompt(call)
        call.step(d("20260811002#"))
        assert call.step(d("9")) == [HOLD]
        assert not call.ended

    def test_silence_at_prompt_reprompts(self, call):
        out = call.step(WAIT)
        assert out[0] == "We did not receive an entry."

    def test_post_menu_repeat_and_end(self, call):
        walk_to_claim_prompt(call)
        call.step(d("20260811001#"))
        call.step(WAIT)
        call.step(WAIT)
        assert "Status: paid" in call.step(d("1"))[0]
        assert call.step(d("9"))[0].endswith("Goodbye.")
        assert call.ended and call.step(d("1")) == []

    def test_post_menu_rep_transfer(self, call):
        walk_to_claim_prompt(call)
        call.step(d("20260811001#"))
        call.step(WAIT)
        call.step(WAIT)
        assert "representative" in call.step(d("2"))[0]

    def test_unknown_claim(self, call):
        walk_to_claim_prompt(call)
        call.step(d("20269999999#"))
        call.step(WAIT)
        assert "could not find" in call.step(WAIT)[0]

    def test_flaky_member_entry_fails_exactly_once(self):
        c = MockInsurerCall(load_db(), flaky=True)
        c.step(d("2"))
        c.step(d("1234567890#"))
        assert "didn't get that" in c.step(d("123456789#"))[0]
        assert "date of birth" in c.step(d("123456789#"))[0]


class TestRep:
    def connect(self, c):
        c.step(d("0"))
        c.step(WAIT)
        return c.step(WAIT)

    def test_speech_can_request_rep(self, call):
        assert "hold" in call.step(say("representative"))[0]

    def test_rep_conversation(self, call):
        assert "Maria" in self.connect(call)[0]
        assert "member ID" in call.step(say("Our NPI is one two three four five six seven eight nine zero"))[0]
        assert "Which claim" in call.step(say("member 123456789 dob 04/12/1985"))[0]
        call.step(say("claim 20260811001"))
        out = call.step(WAIT)
        assert "It is paid" in out[0] and "anything else" in out[1]
        assert "Goodbye" in call.step(say("No, that's all"))[0] and call.ended

    def test_rep_rejects_wrong_identity(self, call):
        self.connect(call)
        assert "repeat the N P I" in call.step(say("hi"))[0]
        call.step(say("npi 1234567890"))
        assert "can't verify" in call.step(say("member 123456789 dob 01/01/2000"))[0]

    def test_rep_cross_member_dob_is_rejected(self, call):
        """Member A's ID with member B's DOB must not verify."""
        self.connect(call)
        call.step(say("npi 1234567890"))
        assert "can't verify" in call.step(say("member 123456789 dob 11/30/2001"))[0]

    def test_rep_asks_again_if_not_done(self, call):
        self.connect(call)
        call.step(say("npi 1234567890"))
        call.step(say("member 123456789 dob 04121985"))
        call.step(say("claim 20260811001"))
        call.step(WAIT)
        assert "anything else" in call.step(say("actually, one question"))[0]
        assert not call.ended


@settings(max_examples=60, deadline=None)
@given(
    st.lists(
        st.one_of(
            st.builds(lambda x: {"type": "dtmf", "digits": x}, st.text(alphabet="0123456789#*", max_size=14)),
            st.builds(lambda x: {"type": "say", "text": x}, st.text(max_size=60)),
            st.just({"type": "wait"}),
        ),
        max_size=60,
    )
)
def test_state_machine_survives_arbitrary_input(actions):
    """Fuzz: whatever the agent sends, the fake payer never crashes and always returns a list of strings."""
    c = MockInsurerCall(load_db())
    c.open()
    for a in actions:
        out = c.step(a)
        assert isinstance(out, list) and all(isinstance(x, str) for x in out)


def test_bundled_data_loads_via_package_resources(tmp_path):
    """Regression: data must ship inside the wheel (it used to live outside the package)."""
    from importlib import resources

    assert resources.files("csva").joinpath("data/claims.json").is_file()
    assert resources.files("csva").joinpath("data/scenarios.json").is_file()
    db = load_db()
    assert "20260811001" in db["claims"]
    custom = tmp_path / "c.json"
    custom.write_text('{"providers": {}, "members": {}, "claims": {}}')
    assert load_db(custom)["claims"] == {}
