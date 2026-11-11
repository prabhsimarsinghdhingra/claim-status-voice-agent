from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

from .agent import ClaimStatusAgent
from .benchmark import load_scenarios, run_benchmark, to_markdown
from .brains import Brain, GeminiBrain, LLMConfigError, RuleBrain
from .db import connect, save_call
from .diagnostics import doctor
from .mock_insurer import MockInsurerCall, load_db
from .models import ClaimQuery, Turn

C = {"insurer": "\033[36m", "agent": "\033[32m", "dim": "\033[2m", "off": "\033[0m", "bold": "\033[1m"}
DEFAULT_RPM = float(os.getenv("CSVA_RPM", "8"))  # stay under free-tier limits; override with --rpm


def make_brain(name: str, rpm: Optional[float] = None) -> Brain:
    if name == "gemini":
        try:
            return GeminiBrain(rpm=rpm if rpm is not None else DEFAULT_RPM)
        except LLMConfigError as exc:
            raise SystemExit(f"error: {exc}") from exc
    return RuleBrain()


def cmd_run(a: argparse.Namespace) -> int:
    scs = {s["id"]: s for s in load_scenarios()}
    if a.scenario not in scs:
        print(f"unknown scenario '{a.scenario}'. choose from: {', '.join(scs)}")
        return 2
    sc = scs[a.scenario]
    q = ClaimQuery(**sc["query"])

    def show(t: Turn) -> None:
        if a.delay:
            time.sleep(a.delay)
        who = "PAYER" if t.speaker == "insurer" else "AGENT"
        col = C[t.speaker]
        txt = f"{C['dim']}{t.text}{C['off']}" if t.text == "[hold music]" else t.text
        print(f"{C['dim']}{t.t_ms:>6}ms{C['off']} {col}{C['bold']}{who:<5}{C['off']} {txt}")

    print(f"{C['bold']}Calling Acme Health provider services - claim {q.claim_number} ({a.brain} brain){C['off']}\n")
    brain = make_brain(a.brain, a.rpm)
    try:
        rec = ClaimStatusAgent(brain).run(q, MockInsurerCall(load_db(), flaky=sc["flaky"]), a.scenario, on_turn=show)
    except LLMConfigError as exc:
        print(f"\nerror: {exc}\nrun `csva doctor` to diagnose.")
        return 3
    print("\n" + C["bold"] + "STRUCTURED RESULT" + C["off"])
    print(rec.result.model_dump_json(indent=2) if rec.result else "no result")
    ms = rec.decision_ms
    print(
        f"\nsuccess={rec.success}  turns={rec.turns}  duration={rec.duration_ms}ms  "
        f"decision latency: mean {sum(ms) / max(len(ms), 1):.1f}ms, max {max(ms, default=0):.1f}ms"
    )
    if getattr(brain, "fallbacks_used", 0):
        print(f"WARNING: {brain.fallbacks_used} decisions fell back to the rule brain (API error / bad output)")  # type: ignore[attr-defined]
    if a.save:
        con = connect(a.db)
        print(f"saved -> {a.db} (call id {save_call(con, rec, q)}, transcript redacted)")
    return 0 if rec.success else 1


def cmd_benchmark(a: argparse.Namespace) -> int:
    only = a.only.split(",") if a.only else None
    try:
        r = run_benchmark(lambda: make_brain(a.brain, a.rpm), repeats=a.repeats, only=only)
    except LLMConfigError as exc:
        print(f"error: {exc}\nrun `csva doctor` to diagnose.")
        return 3
    except ValueError as exc:
        print(f"error: {exc}")
        return 2
    md = to_markdown(a.brain, r)
    print(md)
    if a.out:
        with open(a.out + ".md", "w") as f:
            f.write(md)
        with open(a.out + ".json", "w") as f:
            json.dump(r, f, indent=2)
    if r["call_success_rate"] < a.min_success or r["field_accuracy"] < a.min_accuracy:
        print(f"FAIL: below thresholds (success>={a.min_success}, accuracy>={a.min_accuracy})")
        return 1
    return 0


def cmd_doctor(_: argparse.Namespace) -> int:
    ok_all = True
    for ok, msg in doctor():
        print(("[ ok ] " if ok else "[FAIL] ") + msg)
        ok_all &= ok
    return 0 if ok_all else 1


def cmd_serve(a: argparse.Namespace) -> int:  # pragma: no cover - starts a server
    import uvicorn

    uvicorn.run("csva.twilio_server:app", host="0.0.0.0", port=a.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="csva", description="Claim-status voice agent")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run one simulated call")
    r.add_argument("--scenario", default="ivr_denied_medical_necessity")
    r.add_argument("--brain", choices=["rules", "gemini"], default="rules")
    r.add_argument("--rpm", type=float, default=None, help="max LLM requests/minute (free tier is rate-limited)")
    r.add_argument("--delay", type=float, default=0.0, help="seconds between lines (use ~0.5 when screen-recording)")
    r.add_argument("--save", action="store_true")
    r.add_argument("--db", default="calls.db")
    r.set_defaults(fn=cmd_run)

    b = sub.add_parser("benchmark", help="run scenarios and score them")
    b.add_argument("--brain", choices=["rules", "gemini"], default="rules")
    b.add_argument("--repeats", type=int, default=1)
    b.add_argument("--only", default=None, help="comma-separated scenario ids (saves free-tier quota)")
    b.add_argument("--rpm", type=float, default=None)
    b.add_argument("--out", default=None, help="path prefix, writes .md and .json")
    b.add_argument("--min-success", type=float, default=0.0, help="exit 1 if call success rate is below this")
    b.add_argument("--min-accuracy", type=float, default=0.0, help="exit 1 if field accuracy is below this")
    b.set_defaults(fn=cmd_benchmark)

    d = sub.add_parser("doctor", help="diagnose the Gemini setup (key, model, test call)")
    d.set_defaults(fn=cmd_doctor)

    s = sub.add_parser("serve", help="start the Twilio webhook server")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    return int(a.fn(a))


if __name__ == "__main__":
    sys.exit(main())
