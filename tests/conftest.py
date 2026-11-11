"""Shared fixtures: scenario runner + a scriptable fake Gemini client (no network, no SDK needed)."""

from __future__ import annotations

import json
from collections.abc import Callable
from types import SimpleNamespace

import pytest

from csva.agent import ClaimStatusAgent
from csva.benchmark import load_scenarios
from csva.brains import GeminiBrain, RuleBrain
from csva.mock_insurer import MockInsurerCall, load_db
from csva.models import ClaimQuery, Turn

SCEN = {s["id"]: s for s in load_scenarios()}


def make_call(sid: str) -> tuple[ClaimQuery, MockInsurerCall]:
    sc = SCEN[sid]
    return ClaimQuery(**sc["query"]), MockInsurerCall(load_db(), flaky=sc["flaky"])


def run_scenario(sid: str, brain=None, max_turns: int = 40, on_turn=None):
    q, call = make_call(sid)
    return q, ClaimStatusAgent(brain or RuleBrain(), max_turns=max_turns).run(q, call, sid, on_turn=on_turn)


def transcript_from_prompt(prompt: str) -> list[Turn]:
    body = prompt.split("CALL SO FAR:\n", 1)[1].split("\n\nYour next action", 1)[0]
    turns = []
    for line in body.split("\n"):
        who, text = line.split(": ", 1)
        turns.append(Turn(speaker="insurer" if who == "PAYER" else "agent", text=text))
    return turns


class HttpError(Exception):
    """Mimics google.genai.errors.APIError (has .code)."""

    def __init__(self, code: int, msg: str = "boom"):
        super().__init__(f"{code} {msg}")
        self.code = code


class FakeGemini:
    """Stand-in for genai.Client. `responder(prompt, n)` returns text, None, or raises."""

    def __init__(self, responder: Callable[[str, int], str | None], model_names: list[str] | None = None):
        self.responder = responder
        self.calls: list[dict] = []
        self.models = SimpleNamespace(generate_content=self._generate, list=lambda: self._list(model_names))

    def _generate(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        text = self.responder(contents, len(self.calls) - 1)
        return SimpleNamespace(text=text)

    @staticmethod
    def _list(names):
        return [SimpleNamespace(name=f"models/{n}", supported_actions=["generateContent"]) for n in (names or [])]


def oracle(q: ClaimQuery, include_result: bool = True, mutate: Callable | None = None):
    """A 'perfect model': answers with what RuleBrain would do, as JSON. `mutate(action_dict)` lets tests corrupt it."""
    rb = RuleBrain()

    def respond(prompt: str, _n: int) -> str:
        action = rb.decide(q, transcript_from_prompt(prompt))
        d = json.loads(action.model_dump_json(exclude=None if include_result else {"result"}))
        if mutate:
            d = mutate(d) or d
        return json.dumps(d)

    return respond


@pytest.fixture
def no_sleep():
    slept: list[float] = []
    return slept, slept.append


@pytest.fixture
def gemini_factory(no_sleep):
    _, sleep = no_sleep

    def make(responder, **kw) -> tuple[GeminiBrain, FakeGemini]:
        client = FakeGemini(responder)
        brain = GeminiBrain(model="test-model", client=client, sleep=sleep, **kw)
        return brain, client

    return make
