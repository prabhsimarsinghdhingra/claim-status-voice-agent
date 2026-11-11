import json
import sqlite3

import pytest

from csva import cli
from csva.benchmark import pct, run_benchmark, to_markdown
from csva.brains import GeminiBrain, RuleBrain
from csva.db import connect, save_call
from csva.diagnostics import doctor
from csva.models import Action

from .conftest import SCEN, FakeGemini, run_scenario


# ------------------------------------------------------------------ db
def test_db_roundtrip_and_phi_never_stored(tmp_path):
    q, rec = run_scenario("rep_denied")
    con = connect(tmp_path / "t.db")
    save_call(con, rec, q)
    status, code, ref, blob, rj = con.execute(
        "select status, denial_code, reference_number, transcript_json, result_json from calls"
    ).fetchone()
    assert (status, code) == ("denied", "CO-50") and ref.startswith("REF-")
    for secret in ("123456789", "04121985", "04/12/1985", "JANE DOE", "Jane Doe"):
        assert secret not in blob and secret not in rj


def test_db_failed_call_is_stored_without_result(tmp_path):
    q, rec = run_scenario(
        "ivr_paid", brain=type("M", (), {"name": "mute", "decide": lambda s, q, t: Action(type="wait")})(), max_turns=3
    )
    con = connect(tmp_path / "t.db")
    save_call(con, rec, q)
    assert con.execute("select success, status from calls").fetchone() == (0, None)


def test_db_schema_is_idempotent(tmp_path):
    connect(tmp_path / "t.db").close()
    con = connect(tmp_path / "t.db")
    assert con.execute("select count(*) from calls").fetchone() == (0,)
    assert isinstance(con, sqlite3.Connection)


# ------------------------------------------------------------------ benchmark
def test_benchmark_rules_all_pass():
    r = run_benchmark(RuleBrain)
    assert r["calls"] == len(SCEN) and r["call_success_rate"] == 1.0 and r["field_accuracy"] == 1.0


def test_benchmark_only_filter_and_unknown_filter():
    assert run_benchmark(RuleBrain, only=["ivr_paid"])["calls"] == 1
    with pytest.raises(ValueError):
        run_benchmark(RuleBrain, only=["nope"])


def test_benchmark_can_actually_fail():
    """A gate that can't fail is worthless: a broken brain must score 0."""

    class Broken:
        name = "broken"

        def decide(self, q, t):
            return Action(type="dtmf", digits="9")

    r = run_benchmark(Broken)
    assert r["call_success_rate"] == 0.0 and r["field_accuracy"] < 1.0


def test_benchmark_markdown_aggregates_and_warns():
    r = run_benchmark(RuleBrain, repeats=3)
    md = to_markdown("rules", r)
    assert md.count("| ivr_paid |") == 1 and "| 3 | 3/3 |" in md
    r["llm_fallbacks_to_rules"] = 4
    assert "WARNING" in to_markdown("gemini", r)


def test_percentile_helper():
    assert pct([], 50) == 0.0 and pct([5], 95) == 5 and pct([1, 2, 3, 4, 5], 50) == 3


# ------------------------------------------------------------------ cli
def test_cli_run_ok(capsys):
    assert cli.main(["run", "--scenario", "ivr_paid"]) == 0
    out = capsys.readouterr().out
    assert "STRUCTURED RESULT" in out and '"status": "paid"' in out


def test_cli_run_saves_to_db(tmp_path, capsys):
    db = tmp_path / "c.db"
    assert cli.main(["run", "--scenario", "rep_paid", "--save", "--db", str(db)]) == 0
    assert connect(db).execute("select status from calls").fetchone() == ("paid",)


def test_cli_unknown_scenario(capsys):
    assert cli.main(["run", "--scenario", "zzz"]) == 2


def test_cli_gemini_without_key_refuses(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="GEMINI_API_KEY"):
        cli.main(["run", "--brain", "gemini"])


def test_cli_benchmark_writes_files(tmp_path, capsys):
    out = tmp_path / "bench"
    assert cli.main(["benchmark", "--out", str(out), "--only", "ivr_paid,rep_paid"]) == 0
    assert json.loads((tmp_path / "bench.json").read_text())["calls"] == 2
    assert "Benchmark" in (tmp_path / "bench.md").read_text()


def test_cli_benchmark_threshold_gate(monkeypatch, capsys):
    class Broken:
        name = "broken"

        def decide(self, q, t):
            return Action(type="wait")

    monkeypatch.setattr(cli, "make_brain", lambda name, rpm=None: Broken())
    assert cli.main(["benchmark", "--only", "ivr_paid", "--min-success", "1.0"]) == 1


def test_cli_benchmark_bad_filter(capsys):
    assert cli.main(["benchmark", "--only", "nope"]) == 2


def test_cli_fatal_llm_error_returns_3(monkeypatch, capsys):
    from csva.brains import LLMConfigError

    class Rejecting:
        name = "x"

        def decide(self, q, t):
            raise LLMConfigError("HTTP 401")

    monkeypatch.setattr(cli, "make_brain", lambda name, rpm=None: Rejecting())
    assert cli.main(["run", "--brain", "gemini"]) == 3
    assert cli.main(["benchmark", "--brain", "gemini"]) == 3


def test_cli_doctor_without_key(monkeypatch, capsys):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert cli.main(["doctor"]) == 1
    assert "aistudio.google.com" in capsys.readouterr().out


# ------------------------------------------------------------------ doctor
def brain_with(models, responder=lambda p, n: '{"ok": true}', model="gem-1"):
    return GeminiBrain(model=model, client=FakeGemini(responder, model_names=models))


def test_doctor_all_green():
    checks = doctor(brain_with(["gem-1", "gem-2"]))
    assert all(ok for ok, _ in checks) and len(checks) == 2


def test_doctor_suggests_models_when_configured_one_is_missing():
    (ok, msg), *_ = doctor(brain_with(["gem-flash-a", "gem-flash-b"], model="retired-model"))
    assert not ok and "gem-flash-a" in msg


def test_doctor_reports_failed_test_call():
    def bad(p, n):
        raise RuntimeError("quota")

    checks = doctor(brain_with(["gem-1"], responder=bad))
    assert checks[-1][0] is False and "quota" in checks[-1][1]


def test_doctor_reports_list_failure():
    brain = brain_with(["gem-1"])
    brain.client.models.list = lambda: (_ for _ in ()).throw(RuntimeError("bad key"))
    checks = doctor(brain)
    assert len(checks) == 1 and not checks[0][0] and "bad key" in checks[0][1]


# ------------------------------------------------------------------ the benchmark gate itself
def test_scenario_expectations_match_database_ground_truth():
    """Expected values are derived from the payer DB, so the benchmark isn't grading itself on a hand-picked subset."""
    from csva.mock_insurer import call_ref, load_db

    db = load_db()
    for sid, sc in SCEN.items():
        claim = db["claims"].get(sc["query"]["claim_number"])
        exp = sc["expect"]
        if claim is None:
            assert exp["status"] == "not_found", sid
            continue
        assert exp["status"] == claim["status"], sid
        assert exp["billed_amount"] == claim["billed"] and exp["paid_amount"] == claim["paid"], sid
        assert exp["reference_number"] == call_ref(sc["query"]["claim_number"]), sid


@pytest.mark.parametrize(
    "field",
    ["billed_amount", "paid_amount", "reference_number", "denial_code", "denial_reason", "check_number", "next_action"],
)
def test_benchmark_gate_catches_a_regression_in_every_extracted_field(monkeypatch, field):
    """Mutation test: sabotage one field of the extractor; the gate must notice. (It once missed billed_amount.)"""
    import csva.brains as brains

    real = brains.parse_claim_status

    def sabotaged(text):
        r = real(text)
        if r is not None and getattr(r, field) is not None:
            r = r.model_copy(update={field: None if field != "next_action" else "none"})
        return r

    monkeypatch.setattr(brains, "parse_claim_status", sabotaged)
    result = run_benchmark(RuleBrain)
    assert result["field_accuracy"] < 1.0, f"gate is blind to {field}"
