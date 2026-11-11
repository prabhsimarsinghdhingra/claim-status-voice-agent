"""Deterministic extraction of claim fields from what the payer said (STT text)."""

from __future__ import annotations

import re
from typing import Optional, cast

from .models import ClaimStatusResult, NextAction, Status

_WORD_DIGITS = {
    "zero": "0",
    "oh": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}


def digits_from_text(text: str) -> str:
    """'one two 3 4' / '04/12/1985' -> digits only. Handles spoken digits from STT."""
    out = []
    for tok in re.findall(r"[A-Za-z]+|\d+", text.lower()):
        if tok.isdigit():
            out.append(tok)
        elif tok in _WORD_DIGITS:
            out.append(_WORD_DIGITS[tok])
    return "".join(out)


def _money(m: Optional[re.Match]) -> Optional[float]:
    return float(m.group(1).replace(",", "")) if m else None


def parse_claim_status(text: str) -> Optional[ClaimStatusResult]:
    """Return a result if `text` contains an actual claim outcome, else None.

    The menu line 'For claim status, press 2' must NOT match.
    """
    t = " ".join(text.split())
    low = t.lower()

    if re.search(r"could not find|couldn't find|not finding that claim|no record", low):
        return ClaimStatusResult(status="not_found", next_action="verify_claim_number")

    m = re.search(r"status[:\s]+(?:is\s+)?(paid|denied|pending)", low) or re.search(
        r"\b(?:was|been|is)\s+(paid|denied|pending)\b", low
    )
    if not m:
        return None
    status = cast(Status, m.group(1))

    code = reason = None
    cm = re.search(r"denial code\s*(?:is\s*|:)?\s*([A-Z]{2})[- ]?(\d{1,3})[:,]?\s*([^.]+)\.", t, re.I)
    if cm:
        code, reason = f"{cm.group(1).upper()}-{cm.group(2)}", cm.group(3).strip()
    ref = re.search(r"reference number\s*(?:is\s*|:)?\s*(REF-\d+)", t)
    trace = re.search(r"(?:trace|check) number\s*(?:is\s*|:)?\s*([A-Z0-9]{6,})", t)
    billed = re.search(r"billed amount(?: was| is)?[:\s]*\$([\d,]+(?:\.\d{1,2})?)", t, re.I)
    paid = re.search(r"paid amount(?: was| is)?[:\s]*\$([\d,]+(?:\.\d{1,2})?)", t, re.I)

    if status == "denied":
        if "appeal" in low:
            nxt = "appeal"
        elif code == "CO-16" or "lacks information" in low or "missing" in low:
            nxt = "resubmit"
        else:
            nxt = "none"
    elif status == "pending":
        nxt = "follow_up"
    else:
        nxt = "none"

    return ClaimStatusResult(
        status=status,
        denial_code=code,
        denial_reason=reason,
        billed_amount=_money(billed),
        paid_amount=_money(paid),
        check_number=trace.group(1) if trace else None,
        reference_number=ref.group(1) if ref else None,
        next_action=cast(NextAction, nxt),
    )


# --------------------------------------------------------------------------- #
# Guardrails for LLM-extracted results
# --------------------------------------------------------------------------- #
_FIELDS = (
    "status",
    "denial_code",
    "denial_reason",
    "billed_amount",
    "paid_amount",
    "check_number",
    "reference_number",
    "next_action",
)


def reconcile(model_result: ClaimStatusResult, parsed: ClaimStatusResult) -> ClaimStatusResult:
    """Deterministically-parsed values win wherever the parser found something."""
    merged = model_result.model_copy()
    for f in _FIELDS:
        v = getattr(parsed, f)
        if v not in (None, "unknown", "none"):
            setattr(merged, f, v)
    return merged


def _squash(s: str) -> str:
    return re.sub(r"[^a-z0-9.]", "", s.lower())


def ground(result: ClaimStatusResult, payer_text: str) -> tuple[ClaimStatusResult, list[str]]:
    """Null out any ID / amount the model reports that the payer never actually said.

    Returns (clean_result, names_of_dropped_fields). This is the anti-hallucination check.
    """
    heard = _squash(payer_text)
    out = result.model_copy()
    dropped: list[str] = []

    for f in ("denial_code", "check_number", "reference_number"):
        val = getattr(out, f)
        if val and _squash(str(val)) not in heard:
            setattr(out, f, None)
            dropped.append(f)

    for f in ("billed_amount", "paid_amount"):
        val = getattr(out, f)
        if val is None:
            continue
        candidates = {f"{val:.2f}", f"{val:.1f}", f"{val:g}"}
        if float(val).is_integer():
            candidates.add(str(int(val)))
        if not any(c in heard for c in candidates):
            setattr(out, f, None)
            dropped.append(f)
    return out, dropped
