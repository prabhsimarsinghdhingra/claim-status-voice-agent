"""Decision policies ("brains") for the call agent.

RuleBrain   - deterministic, offline, zero latency. Used as a test harness / baseline.
GeminiBrain - Google Gemini (free tier via AI Studio) reads the live transcript and picks the next
              action (press a key / say something / wait) and extracts the structured result.
              Transient API errors are retried with backoff, then fall back to RuleBrain (counted);
              config errors (bad key / bad model) abort loudly instead of silently degrading.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable
from typing import Optional, Protocol

from .models import Action, ClaimQuery, ClaimStatusResult, Turn
from .parsing import ground, parse_claim_status, reconcile


def heard_since_last_agent_turn(transcript: list[Turn]) -> str:
    chunks: list[str] = []
    for t in reversed(transcript):
        if t.speaker == "agent":
            break
        chunks.append(t.text)
    return " ".join(reversed(chunks))


def fmt_dob(d: str) -> str:
    return f"{d[:2]}/{d[2:4]}/{d[4:]}"


class Brain(Protocol):
    name: str

    def decide(self, query: ClaimQuery, transcript: list[Turn]) -> Action: ...


# --------------------------------------------------------------------------- #
class RuleBrain:
    name = "rules"

    def __init__(self) -> None:
        self.result: Optional[ClaimStatusResult] = None
        self.last: Optional[Action] = None
        self.retries = 0

    def _emit(self, action: Action) -> Action:
        if self.result is not None:
            action.result = self.result
        self.last = action
        return action

    def decide(self, q: ClaimQuery, transcript: list[Turn]) -> Action:
        heard = heard_since_last_agent_turn(transcript)
        low = heard.lower()
        flat = re.sub(r"\bn p i\b", "npi", low)

        # 1) did we just hear the outcome?
        parsed = parse_claim_status(heard)
        if parsed and self.result is None:
            self.result = parsed

        if self.result is not None:
            if "press 9" in low:
                return self._emit(Action(type="dtmf", digits="9"))
            if "anything else" in low:
                return self._emit(Action(type="say", text="No, that's all. Thank you for your help."))
            return self._emit(Action(type="wait"))

        # 2) payer didn't understand us -> resend (bounded)
        if (
            re.search(r"didn.t (get|catch|understand)|not valid|could not verify|try again|did not receive", low)
            and self.last
        ):
            if self.retries < 3 and self.last.type != "wait":
                self.retries += 1
                return self._emit(self.last)
        else:
            self.retries = 0

        # 3) IVR menu
        if "press 2" in low and "claim status" in low:
            return self._emit(Action(type="dtmf", digits="0" if q.prefer_rep else "2"))
        if "enter" in low and "npi" in flat:
            return self._emit(Action(type="dtmf", digits=q.npi + "#"))
        if "enter" in low and "member id" in low:
            return self._emit(Action(type="dtmf", digits=q.member_id + "#"))
        if "enter" in low and "date of birth" in low:
            return self._emit(Action(type="dtmf", digits=q.member_dob))
        if "enter" in low and "claim number" in low:
            return self._emit(Action(type="dtmf", digits=q.claim_number + "#"))

        # 4) live representative
        if "name" in low and "npi" in flat:
            return self._emit(
                Action(
                    type="say",
                    text=(f"Hi Maria, this is Alex calling from {q.provider_name} billing. Our NPI is {q.npi}."),
                )
            )
        if "member id" in low and "date of birth" in low:
            return self._emit(
                Action(type="say", text=(f"Member ID {q.member_id}, date of birth {fmt_dob(q.member_dob)}."))
            )
        if "which claim" in low or "claim number" in low:
            return self._emit(
                Action(
                    type="say",
                    text=(
                        f"Claim number {q.claim_number} for {q.patient_name}. I'm calling to check the claim status."
                    ),
                )
            )

        # 5) hold music / nothing to do
        return self._emit(Action(type="wait"))


# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """You are a voice agent making an outbound phone call to a US health insurer's provider-services line on behalf of a medical billing company. Your single goal: get the status of ONE claim and record it accurately.

You act one step at a time. Each turn you see the call transcript so far and must reply with ONE JSON object and nothing else:
  {"type": "dtmf", "digits": "2"}                 -> press keys (use '#' where the IVR says 'followed by the pound key')
  {"type": "say", "text": "..."}                  -> speak (use with live humans or speech prompts)
  {"type": "wait"}                                -> stay silent (hold music, 'please hold', the payer is still talking)
Optionally add "result" ONLY once the payer has actually told you the claim outcome:
  "result": {"status": "paid|denied|pending|not_found", "denial_code": "CO-50"|null, "denial_reason": str|null,
             "billed_amount": number|null, "paid_amount": number|null, "check_number": str|null,
             "reference_number": str|null, "next_action": "none|appeal|resubmit|follow_up|verify_claim_number"}

Rules:
- Only use facts in the CASE FILE. Never invent IDs, dates or amounts. Only give what is asked, when asked (minimum necessary - this is PHI).
- Navigate to claim status. If prefer_rep is true, press the key for a representative instead and talk to the human.
- Enter IDs exactly as given. If the payer says it didn't understand / invalid entry, re-enter once or twice, then try the other channel (speak instead of keypad, or ask for a representative).
- During hold music or 'please hold', output {"type":"wait"}.
- Copy denial codes in CARC group-code form (CO-50, PR-1...). Copy amounts and reference/trace numbers EXACTLY from what was said.
- next_action: denied with appeal language -> appeal; denied for missing/incorrect info -> resubmit; pending -> follow_up; not found -> verify_claim_number; else none.
- After you have the outcome, end the call politely: press the 'end call' key if offered, otherwise say you need nothing else. Keep attaching the same "result" on that final action.
- Output raw JSON only. No prose, no markdown fences."""


def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        raise ValueError("no JSON object in model output")
    return json.loads(text[start : end + 1])


class LLMConfigError(RuntimeError):
    """Bad API key / unknown model / missing SDK. Never silently downgraded to the rule brain."""


TRANSIENT_CODES = {408, 429, 500, 502, 503, 504}
FATAL_CODES = {400, 401, 403, 404}


def status_code_of(exc: BaseException) -> Optional[int]:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code
    m = re.search(r"\b([45]\d\d)\b", str(exc))
    return int(m.group(1)) if m else None


class LLMBrain:
    """Provider-agnostic LLM policy: prompt -> JSON action -> validate -> ground -> (fallback)."""

    name = "llm"

    def __init__(
        self,
        fallback: Optional[Brain] = None,
        rpm: Optional[float] = None,
        max_retries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.fallback = fallback or RuleBrain()
        self.min_interval = 60.0 / rpm if rpm else 0.0
        self.max_retries = max_retries
        self._sleep, self._clock = sleep, clock
        self._last_call = -1e9
        self.result: Optional[ClaimStatusResult] = None
        # stats surfaced by the benchmark
        self.api_calls = 0
        self.retries = 0
        self.fallbacks_used = 0
        self.grounding_drops = 0
        self.last_wait_ms = 0.0  # throttle + backoff time inside the latest decide(); excluded from latency

    # -- provider hook -------------------------------------------------------
    def _complete(self, system: str, user: str) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    # -- plumbing ------------------------------------------------------------
    def _throttle(self) -> None:
        wait = self._last_call + self.min_interval - self._clock()
        if wait > 0:
            self.last_wait_ms += wait * 1000
            self._sleep(wait)
        self._last_call = self._clock()

    def _call_with_retry(self, user: str) -> str:
        for attempt in range(self.max_retries + 1):
            self._throttle()
            self.api_calls += 1
            try:
                return self._complete(SYSTEM_PROMPT, user)
            except Exception as exc:
                code = status_code_of(exc)
                if code in FATAL_CODES:
                    raise LLMConfigError(f"{self.name} API rejected the request (HTTP {code}): {exc}") from exc
                if code in TRANSIENT_CODES and attempt < self.max_retries:
                    self.retries += 1
                    backoff = min(30.0, 2.0 * (2**attempt))
                    self.last_wait_ms += backoff * 1000
                    self._sleep(backoff)
                    continue
                raise
        raise RuntimeError("unreachable")  # pragma: no cover

    @staticmethod
    def _render(q: ClaimQuery, transcript: list[Turn]) -> str:
        lines = [f"{'PAYER' if t.speaker == 'insurer' else 'YOU'}: {t.text}" for t in transcript[-30:]]
        return (
            "CASE FILE:\n"
            + q.model_dump_json(indent=1)
            + "\n\nCALL SO FAR:\n"
            + "\n".join(lines)
            + "\n\nYour next action (JSON):"
        )

    def decide(self, q: ClaimQuery, transcript: list[Turn]) -> Action:
        self.last_wait_ms = 0.0
        try:
            raw = self._call_with_retry(self._render(q, transcript))
            action = Action.model_validate(_extract_json(raw))
        except LLMConfigError:
            raise
        except Exception:
            self.fallbacks_used += 1
            return self.fallback.decide(q, transcript)

        parsed = parse_claim_status(heard_since_last_agent_turn(transcript))
        if action.result is not None and parsed is None and self.result is None:
            action.result = None  # model claims an outcome the payer never stated
        if action.result is not None:
            if parsed is not None:
                action.result = reconcile(action.result, parsed)
            payer_text = " ".join(t.text for t in transcript if t.speaker == "insurer")
            action.result, dropped = ground(action.result, payer_text)
            self.grounding_drops += len(dropped)
        elif parsed is not None:
            action.result = parsed
        if action.result is not None:
            self.result = action.result
        elif self.result is not None:
            action.result = self.result
        return action


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"


def gemini_api_key() -> Optional[str]:
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


class GeminiBrain(LLMBrain):
    """Google Gemini via the `google-genai` SDK (free tier from https://aistudio.google.com/apikey)."""

    name = "gemini"

    def __init__(self, model: Optional[str] = None, client=None, api_key: Optional[str] = None, **kw) -> None:
        super().__init__(**kw)
        self.model = model or os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
        if client is None:
            key = api_key or gemini_api_key()
            if not key:
                raise LLMConfigError("GEMINI_API_KEY is not set (get a free key at https://aistudio.google.com/apikey)")
            try:
                from google import genai  # imported lazily so the repo runs without the SDK
            except ImportError as exc:
                raise LLMConfigError(
                    "google-genai is not installed: pip install 'claim-status-voice-agent[gemini]'"
                ) from exc
            client = genai.Client(api_key=key)
        self.client = client

    @staticmethod
    def generation_config() -> dict:
        # A plain dict is accepted by the SDK (validated against types.GenerateContentConfig in tests).
        return {
            "system_instruction": SYSTEM_PROMPT,
            "temperature": 0,
            "response_mime_type": "application/json",
            "max_output_tokens": 1024,
        }

    def _complete(self, system: str, user: str) -> str:
        resp = self.client.models.generate_content(model=self.model, contents=user, config=self.generation_config())
        text = getattr(resp, "text", None)
        if not text:
            raise ValueError("empty Gemini response (blocked or truncated)")
        return text
