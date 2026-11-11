"""A fake US payer phone system ("Acme Health Provider Services").

Simulates the two things a real claims-status call throws at an agent:
  1. an IVR menu tree (DTMF or speech input, hold music, input errors, re-prompts)
  2. a live representative (free-form conversation, identity verification)

It is entirely synthetic - no real payer, no real PHI.
Each `step()` returns the list of things the payer "said" in response.
"""

from __future__ import annotations

import json
import re
from importlib import resources
from pathlib import Path
from typing import Optional

from .parsing import digits_from_text

HOLD = "[hold music]"
MENU = (
    "For eligibility and benefits, press 1. For claim status, press 2. "
    "For prior authorizations, press 3. To speak with a representative, press 0."
)
P_NPI = "Please enter the ten digit N P I of the billing provider, followed by the pound key."
P_MEMBER = "Please enter the member ID, numbers only, followed by the pound key."
P_DOB = "Please enter the member's date of birth as eight digits: month, day, year."
P_CLAIM = "Please enter the claim number, followed by the pound key."
POST_MENU = "To hear that again, press 1. To speak with a representative, press 2. To end this call, press 9."


def load_db(path: Path | None = None) -> dict:
    """Load the synthetic payer database (bundled with the package unless a path is given)."""
    text = Path(path).read_text() if path else resources.files("csva").joinpath("data/claims.json").read_text()
    return json.loads(text)


def call_ref(claim: str) -> str:
    return f"REF-{(int(claim[-3:]) * 7919) % 90000 + 10000}"


class MockInsurerCall:
    HOLD_TICKS = 2

    def __init__(self, db: Optional[dict] = None, *, flaky: bool = False):
        self.db = db or load_db()
        self.flaky = flaky
        self.ended = False
        self.state = "menu"
        self.ctx: dict = {}
        self._flaked = False
        self._ticks = 0
        self._pending_hold: Optional[str] = None

    # ---------------------------------------------------------------- helpers
    def _result_text(self, claim: str, style: str) -> str:
        c = self.db["claims"].get(claim)
        if not c:
            if style == "ivr":
                return "We could not find a claim with that number on file for this provider."
            return "I'm sorry, I could not find a claim with that number on file."
        member = self.db["members"][c["member"]]["name"]
        ref = call_ref(claim)
        extra = ""
        if c["status"] == "paid":
            extra = f"Payment was issued by {c['method']}, trace number {c['trace']}, on {c['date']}."
        elif c["status"] == "pending":
            extra = f"This claim is still being processed. Expected completion date {c['eta']}."
        elif c.get("code") == "CO-50":
            extra = "You may file an appeal within 180 days of the denial date."
        elif c.get("code") == "CO-16":
            extra = "Please submit a corrected claim."
        denial = f"Denial code {c['code']}: {c['reason']}. " if c["status"] == "denied" else ""
        if style == "ivr":
            return (
                f"Claim number {claim} for patient {member}. Status: {c['status']}. {denial}"
                f"Billed amount ${c['billed']:.2f}. Paid amount ${c['paid']:.2f}. {extra} "
                f"Call reference number {ref}."
            )
        denial_rep = f"The denial code is {c['code']}: {c['reason']}. " if c["status"] == "denied" else ""
        return (
            f"Okay, I have claim {claim} here. It is {c['status']}. {denial_rep}"
            f"The billed amount was ${c['billed']:.2f} and the paid amount is ${c['paid']:.2f}. {extra} "
            f"Your call reference number is {ref}."
        )

    def _start_hold(self, kind: str, ticks: int) -> None:
        self.state = "hold"
        self._pending_hold = kind
        self._ticks = ticks

    def _tick(self) -> list[str]:
        self._ticks -= 1
        if self._ticks > 0:
            return [HOLD]
        kind, self._pending_hold = self._pending_hold, None
        if kind == "lookup":
            self.state = "post"
            return [self._result_text(self.ctx["claim"], "ivr"), POST_MENU]
        if kind == "rep_connect":
            self.state = "rep_npi"
            return [
                "Thank you for holding, this is Maria with Acme Health provider services. "
                "Can I get your name and the N P I?"
            ]
        if kind == "rep_lookup":
            self.state = "rep_end"
            return [self._result_text(self.ctx["claim"], "rep"), "Is there anything else I can help you with today?"]
        raise RuntimeError(kind)

    PROMPTS = {"ask_npi": P_NPI, "ask_member": P_MEMBER, "ask_dob": P_DOB, "ask_claim": P_CLAIM, "menu": MENU}

    # ------------------------------------------------------------------- API
    def open(self) -> list[str]:
        return ["Thank you for calling Acme Health provider services. This call may be recorded. " + MENU]

    def step(self, action: dict) -> list[str]:
        if self.ended:
            return []
        kind = action.get("type", "wait")
        text = (action.get("text") or "") if kind == "say" else ""
        digits = re.sub(r"\D", "", action.get("digits") or "") if kind == "dtmf" else digits_from_text(text)
        low = text.lower()
        s = self.state

        if s == "hold":
            return self._tick()

        if kind == "wait":
            if s in self.PROMPTS:
                return ["We did not receive an entry.", self.PROMPTS[s]]
            return []

        # ---------------- IVR ----------------
        if s == "menu":
            if digits == "2" or "claim" in low:
                self.state = "ask_npi"
                return [P_NPI]
            if digits == "0" or re.search(r"representative|agent|operator", low):
                self._start_hold("rep_connect", self.HOLD_TICKS)
                return ["Please hold for the next available representative.", HOLD]
            if digits in ("1", "3"):
                return ["Sorry, that option is not available in this demo.", MENU]
            return ["Sorry, I didn't get that.", MENU]

        if s == "ask_npi":
            if digits in self.db["providers"]:
                self.ctx["npi"] = digits
                self.state = "ask_member"
                return [P_MEMBER]
            return ["Sorry, I didn't get that.", P_NPI]

        if s == "ask_member":
            if self.flaky and not self._flaked:
                self._flaked = True  # simulate a dropped DTMF / STT miss once
                return ["Sorry, I didn't get that.", P_MEMBER]
            if digits in self.db["members"]:
                self.ctx["member"] = digits
                self.state = "ask_dob"
                return [P_DOB]
            return ["Sorry, that member ID is not valid.", P_MEMBER]

        if s == "ask_dob":
            if digits == self.db["members"][self.ctx["member"]]["dob"]:
                self.state = "ask_claim"
                return [P_CLAIM]
            return ["Sorry, I could not verify that date of birth.", P_DOB]

        if s == "ask_claim":
            if len(digits) >= 8:
                self.ctx["claim"] = digits
                self._start_hold("lookup", self.HOLD_TICKS)
                return ["Thank you. Please hold while I look up that claim.", HOLD]
            return ["Sorry, I didn't get that.", P_CLAIM]

        if s == "post":
            if digits == "9":
                self.ended = True
                return ["Thank you for calling Acme Health. Goodbye."]
            if digits == "1":
                return [self._result_text(self.ctx["claim"], "ivr"), POST_MENU]
            if digits == "2":
                self._start_hold("rep_connect", self.HOLD_TICKS)
                return ["Please hold for the next available representative.", HOLD]
            return ["Sorry, I didn't get that.", POST_MENU]

        # ---------------- live rep ----------------
        if s == "rep_npi":
            if self.ctx.get("npi_ok") or any(p in digits for p in self.db["providers"]):
                self.state = "rep_member"
                return ["Thanks. Can I get the member ID and date of birth?"]
            return ["Sorry, could you repeat the N P I?"]

        if s == "rep_member":
            for mid, m in self.db["members"].items():
                if mid in digits and m["dob"] in digits:
                    self.ctx["member"] = mid
                    self.state = "rep_claim"
                    return ["Thank you, I've verified the patient. Which claim are you calling about?"]
            return ["I'm sorry, I can't verify that. Could you repeat the member ID and date of birth?"]

        if s == "rep_claim":
            m = re.search(r"\d{8,}", digits)
            if m:
                self.ctx["claim"] = m.group(0)
                self._start_hold("rep_lookup", 1)
                return ["One moment while I pull that up."]
            return ["Sorry, could you repeat the claim number?"]

        if s == "rep_end":
            if re.search(r"\bno\b|that'?s all|nothing", low):
                self.ended = True
                return ["Have a great day. Goodbye."]
            return ["Is there anything else I can help you with today?"]

        return []
