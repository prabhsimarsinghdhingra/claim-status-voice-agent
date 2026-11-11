from __future__ import annotations

import json
import statistics
from importlib import resources

from .agent import ClaimStatusAgent
from .mock_insurer import MockInsurerCall, load_db
from .models import CallRecord, ClaimQuery


def load_scenarios() -> list[dict]:
    return json.loads(resources.files("csva").joinpath("data/scenarios.json").read_text())


def score(rec: CallRecord, expect: dict) -> tuple[int, int]:
    """Field-level score: (fields correct, fields expected)."""
    got = rec.result.model_dump() if rec.result else {}
    ok = 0
    for k, v in expect.items():
        g = got.get(k)
        ok += int(abs(g - v) < 0.005) if isinstance(v, float) and g is not None else int(g == v)
    return ok, len(expect)


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(p / 100 * (len(xs) - 1)))]


def run_benchmark(make_brain, repeats: int = 1, only: list[str] | None = None) -> dict:
    db = load_db()
    rows, all_ms, fallbacks, drops, api_calls = [], [], 0, 0, 0
    f_ok = f_tot = calls_ok = 0
    scenarios = [sc for sc in load_scenarios() if not only or sc["id"] in only]
    if not scenarios:
        raise ValueError(f"no scenarios match {only}")
    for sc in scenarios:
        for _ in range(repeats):
            q = ClaimQuery(**sc["query"])
            brain = make_brain()
            rec = ClaimStatusAgent(brain).run(q, MockInsurerCall(db, flaky=sc["flaky"]), sc["id"])
            fallbacks += getattr(brain, "fallbacks_used", 0)
            drops += getattr(brain, "grounding_drops", 0)
            api_calls += getattr(brain, "api_calls", 0)
            ok, tot = score(rec, sc["expect"])
            f_ok, f_tot = f_ok + ok, f_tot + tot
            passed = rec.success and ok == tot
            calls_ok += int(passed)
            all_ms += rec.decision_ms
            rows.append(
                {
                    "scenario": sc["id"],
                    "passed": passed,
                    "fields": f"{ok}/{tot}",
                    "turns": rec.turns,
                    "mean_ms": round(statistics.mean(rec.decision_ms), 1) if rec.decision_ms else 0,
                }
            )
    n = len(rows)
    return {
        "calls": n,
        "call_success_rate": calls_ok / n,
        "field_accuracy": f_ok / f_tot,
        "decision_ms_p50": round(pct(all_ms, 50), 1),
        "decision_ms_p95": round(pct(all_ms, 95), 1),
        "mean_turns": round(statistics.mean(r["turns"] for r in rows), 1),
        "llm_fallbacks_to_rules": fallbacks,
        "grounding_drops": drops,
        "api_calls": api_calls,
        "rows": rows,
    }


def to_markdown(brain_name: str, r: dict) -> str:
    out = [
        f"### Benchmark - `{brain_name}` brain vs. mock insurer\n",
        f"- Calls: **{r['calls']}**  |  Call success: **{r['call_success_rate']:.0%}**"
        f"  |  Field-level accuracy: **{r['field_accuracy']:.0%}**",
        f"- Decision latency: p50 **{r['decision_ms_p50']} ms**, p95 **{r['decision_ms_p95']} ms**"
        f"  |  Mean turns/call: **{r['mean_turns']}**",
    ]
    if r.get("api_calls"):
        out.append(
            f"- LLM API calls: **{r['api_calls']}**"
            f"  |  Values dropped by the grounding check: **{r.get('grounding_drops', 0)}**"
        )
    if r.get("llm_fallbacks_to_rules"):
        out.append(
            f"- WARNING: **{r['llm_fallbacks_to_rules']}** decisions fell back to the rule brain"
            " (API error / bad JSON) - not pure LLM results."
        )
    out += ["", "| Scenario | Runs | Passed | Fields | Mean turns | Mean decision ms |", "|---|---|---|---|---|---|"]
    by: dict = {}
    for x in r["rows"]:
        by.setdefault(x["scenario"], []).append(x)
    for sid, xs in by.items():
        ok = sum(1 for x in xs if x["passed"])
        out.append(
            f"| {sid} | {len(xs)} | {ok}/{len(xs)} | {xs[0]['fields']} | "
            f"{statistics.mean(x['turns'] for x in xs):.1f} | {statistics.mean(x['mean_ms'] for x in xs):.1f} |"
        )
    return "\n".join(out) + "\n"
