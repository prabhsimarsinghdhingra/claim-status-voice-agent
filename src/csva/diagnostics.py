"""`csva doctor`: figure out in 10 seconds why the Gemini brain isn't working."""

from __future__ import annotations

from .brains import GeminiBrain, gemini_api_key


def doctor(brain: GeminiBrain | None = None) -> list[tuple[bool, str]]:
    checks: list[tuple[bool, str]] = []
    key = gemini_api_key()
    if brain is None:
        checks.append(
            (
                bool(key),
                "GEMINI_API_KEY set"
                if key
                else "GEMINI_API_KEY missing -> create a free key at https://aistudio.google.com/apikey",
            )
        )
        if not key:
            return checks
        try:
            brain = GeminiBrain()
        except Exception as exc:
            checks.append((False, f"cannot create client: {exc}"))
            return checks

    try:
        names = []
        for m in brain.client.models.list():
            actions = getattr(m, "supported_actions", None) or ["generateContent"]
            if "generateContent" in actions:
                names.append(str(getattr(m, "name", "")).removeprefix("models/"))
        found = brain.model in names
        flash = sorted(n for n in names if "flash" in n)[:8]
        checks.append(
            (
                found,
                f"model '{brain.model}' "
                + (
                    "is available"
                    if found
                    else f"NOT available to this key. Try GEMINI_MODEL=<one of: {', '.join(flash)}>"
                ),
            )
        )
    except Exception as exc:
        checks.append((False, f"could not list models (bad key, network, or quota?): {exc}"))
        return checks

    try:
        out = brain._complete("", 'Reply with exactly this JSON and nothing else: {"ok": true}')
        checks.append(("ok" in out.lower(), f"test call returned: {out.strip()[:60]!r}"))
    except Exception as exc:
        checks.append((False, f"test call failed: {exc}"))
    return checks
