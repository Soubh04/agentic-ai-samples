"""The Jev side, tested against the real typesafe_sdk classes. No network: HTTP is replaced by a mock transport."""
import json
import os

import httpx2
import pytest
from helpers import jev_payload, jev_response
from typesafe_sdk import (
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
    SystemOneResponse,
    TypeSafeAPIError,
    TypeSafeAuthenticationError,
    TypeSafeClient,
    TypeSafeError,
)

from core import TEAMS, URGENCY, BackendError, CallStats, cost_of
from jev_backend import QUESTIONS, JevBackend, to_decision


def test_questions_are_built_from_the_real_sdk_classes():
    assert isinstance(QUESTIONS["team"], Choice)
    assert isinstance(QUESTIONS["urgency"], Score)
    assert isinstance(QUESTIONS["refund_request"], Noul)
    assert list(QUESTIONS["team"].criteria) == list(TEAMS)
    # Score criteria are ordered: the position is the level, so index 2 must be the most urgent.
    assert list(QUESTIONS["urgency"].criteria) == list(URGENCY.values())
    assert QUESTIONS["urgency"].criteria[-1] == URGENCY["today"]


def test_questions_serialize_to_the_wire_form():
    assert QUESTIONS["team"].model_dump()["type"] == "choice"
    assert QUESTIONS["urgency"].model_dump()["type"] == "score"
    assert QUESTIONS["refund_request"].model_dump()["type"] == "noul"


def test_adapter_reads_a_real_sdk_response():
    response = jev_response(confidence=0.93, odds={"0": 0.05, "1": 0.15, "2": 0.80}, refund=0.97)
    assert isinstance(response, SystemOneResponse)
    decision = to_decision(response, latency_ms=150.0)
    assert (decision.team, decision.urgency, decision.refund_request) == ("billing", "today", True)
    assert decision.team_confidence == 0.93
    assert decision.team_probabilities["billing"] == 0.93
    assert decision.refund_probability == 0.97
    assert decision.stats == CallStats("jev-1.13.0", 150.0, 118, 12, cost_of("typesafe-jev", 118, 12))


def test_both_documented_access_paths_exist():
    response = jev_response()
    assert response.answers["team"].choice == response.choices["team"].choice == "billing"
    assert response.answers["urgency"].score == response.scores["urgency"].score
    assert response.answers["refund_request"].noul == response.nouls["refund_request"].noul
    assert isinstance(response.answers["team"], ChoiceAnswer)
    assert isinstance(response.answers["urgency"], ScoreAnswer)
    assert isinstance(response.answers["refund_request"], NoulAnswer)
    # `answers` mixes answer types, which is why the adapter reads the typed views instead.
    with pytest.raises(AttributeError):
        response.answers["refund_request"].choice  # noqa: B018


def test_urgency_is_the_most_likely_level_not_the_rounded_expected_score():
    # 45% "can wait" and 55% "today": the expected score is 1.1, which would round to "this week",
    # a level the model gave 0% probability.
    response = jev_response(odds={"0": 0.45, "1": 0.0, "2": 0.55})
    assert response.scores["urgency"].score == pytest.approx(1.1)
    assert to_decision(response, 1.0).urgency == "today"


def test_an_urgency_tie_goes_to_the_more_urgent_level():
    assert to_decision(jev_response(odds={"0": 0.5, "1": 0.0, "2": 0.5}), 1.0).urgency == "today"


def test_refund_is_yes_at_one_half_and_above():
    assert to_decision(jev_response(refund=0.5), 1.0).refund_request is True
    assert to_decision(jev_response(refund=0.49), 1.0).refund_request is False


def test_a_missing_answer_is_a_clear_error():
    payload = jev_payload()
    del payload["answers"]["urgency"]
    response = SystemOneResponse.from_http_response(httpx2.Response(200, json=payload))
    with pytest.raises(BackendError, match="urgency"):
        to_decision(response, 1.0)


def test_unreported_token_counts_mean_no_cost_rather_than_a_guess():
    payload = jev_payload()
    payload["usage"] = {}
    stats = to_decision(SystemOneResponse.from_http_response(httpx2.Response(200, json=payload)), 1.0).stats
    assert (stats.input_tokens, stats.output_tokens, stats.cost_usd) == (None, None, None)


def test_the_backend_sends_the_documented_request(fake_keys):
    seen = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.update(path=request.url.path, auth=request.headers["authorization"], body=json.loads(request.content))
        return httpx2.Response(200, json=jev_payload())

    decision = JevBackend(TypeSafeClient(transport=httpx2.MockTransport(handler))).decide("I was charged twice")
    assert seen["path"] == "/v1/systemone"
    assert seen["auth"] == f"Bearer {os.environ['TYPESAFE_API_KEY']}"  # read from the environment
    assert seen["body"]["state"] == "I was charged twice"
    assert seen["body"]["model"] == "jev-latest"
    assert {name: q["type"] for name, q in seen["body"]["questions"].items()} == {
        "team": "choice",
        "urgency": "score",
        "refund_request": "noul",
    }
    assert set(seen["body"]["questions"]["team"]["criteria"]) == set(TEAMS)
    assert decision.team == "billing"
    assert decision.stats.latency_ms >= 0


def test_a_rejected_key_raises_the_sdks_authentication_error(fake_keys):
    def handler(request):
        return httpx2.Response(401, json={"detail": "invalid key"})

    backend = JevBackend(TypeSafeClient(transport=httpx2.MockTransport(handler)))
    with pytest.raises(TypeSafeAuthenticationError):
        backend.decide("anything")
    assert issubclass(TypeSafeAuthenticationError, TypeSafeAPIError)
    assert issubclass(TypeSafeAPIError, TypeSafeError)
