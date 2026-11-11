from __future__ import annotations

import re
import time
from collections.abc import Callable
from typing import Optional

from .brains import Brain
from .models import CallRecord, ClaimQuery, ClaimStatusResult, Speaker, Turn


class ClaimStatusAgent:
    """Runs one call: listen -> decide -> act, until the payer hangs up or we hit max_turns."""

    def __init__(self, brain: Brain, max_turns: int = 40):
        self.brain = brain
        self.max_turns = max_turns

    def run(
        self,
        query: ClaimQuery,
        call,
        scenario_id: str = "adhoc",
        on_turn: Optional[Callable[[Turn], None]] = None,
    ) -> CallRecord:
        t0 = time.perf_counter()
        transcript: list[Turn] = []
        decision_ms: list[float] = []
        result: Optional[ClaimStatusResult] = None

        def log(speaker: Speaker, text: str) -> None:
            turn = Turn(speaker=speaker, text=text, t_ms=int((time.perf_counter() - t0) * 1000))
            transcript.append(turn)
            if on_turn:
                on_turn(turn)

        for line in call.open():
            log("insurer", line)

        for _ in range(self.max_turns):
            if call.ended:
                break
            s = time.perf_counter()
            action = self.brain.decide(query, transcript)
            # latency = model/compute time only; rate-limit throttling and retry backoff are not 'slowness'
            waited = getattr(self.brain, "last_wait_ms", 0.0)
            decision_ms.append(max(0.0, (time.perf_counter() - s) * 1000 - waited))
            if action.result is not None:
                result = action.result
            log("agent", action.describe())
            for line in call.step(action.to_wire()):
                log("insurer", line)

        return CallRecord(
            scenario_id=scenario_id,
            brain=self.brain.name,
            success=bool(result and result.status != "unknown" and call.ended),
            result=result,
            transcript=transcript,
            decision_ms=decision_ms,
            duration_ms=int((time.perf_counter() - t0) * 1000),
            turns=len(decision_ms),
        )


# ---- PHI hygiene: what we persist is redacted ---------------------------------
def redact(text: str, q: ClaimQuery) -> str:
    out = text
    dob = q.member_dob
    for variant in (dob, f"{dob[:2]}/{dob[2:4]}/{dob[4:]}"):
        out = out.replace(variant, "[DOB]")
    out = out.replace(q.member_id, "[MEMBER_ID]")
    out = re.sub(re.escape(q.patient_name), "[PATIENT]", out, flags=re.I)
    return out
