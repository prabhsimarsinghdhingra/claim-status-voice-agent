from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ClaimQuery(BaseModel):
    """Everything the agent is allowed to know / say on the call (the 'case file')."""

    npi: str
    member_id: str
    member_dob: str = Field(description="MMDDYYYY")
    claim_number: str
    patient_name: str
    provider_name: str
    prefer_rep: bool = False  # True = press 0 and talk to a human


Status = Literal["paid", "denied", "pending", "not_found", "unknown"]
NextAction = Literal["none", "appeal", "resubmit", "follow_up", "verify_claim_number"]


class ClaimStatusResult(BaseModel):
    """Structured output saved to the DB after every call."""

    status: Status = "unknown"
    denial_code: Optional[str] = None  # CARC group-code style, e.g. CO-50
    denial_reason: Optional[str] = None
    billed_amount: Optional[float] = None
    paid_amount: Optional[float] = None
    check_number: Optional[str] = None  # EFT / check trace number
    reference_number: Optional[str] = None  # call reference given by payer
    next_action: NextAction = "none"


class Action(BaseModel):
    """What the agent does next. `result` is attached once the claim details were heard."""

    type: Literal["dtmf", "say", "wait"]
    digits: Optional[str] = None
    text: Optional[str] = None
    result: Optional[ClaimStatusResult] = None

    def to_wire(self) -> dict:
        return {"type": self.type, "digits": self.digits or "", "text": self.text or ""}

    def describe(self) -> str:
        if self.type == "dtmf":
            return f"[DTMF {self.digits}]"
        if self.type == "say":
            return self.text or ""
        return "[silence]"


Speaker = Literal["insurer", "agent"]


class Turn(BaseModel):
    speaker: Speaker
    text: str
    t_ms: int = 0


class CallRecord(BaseModel):
    scenario_id: str
    brain: str
    success: bool
    result: Optional[ClaimStatusResult]
    transcript: list[Turn]
    decision_ms: list[float]
    duration_ms: int
    turns: int
