import pytest
from pydantic import ValidationError

from laya_server.schemas import PredictRequest


def test_choice_requires_two_criteria():
    with pytest.raises(ValidationError):
        PredictRequest(
            state="hello",
            questions={
                "intent": {
                    "type": "choice",
                    "instructions": "Pick one",
                    "criteria": ["only-one"],
                }
            },
        )


def test_valid_mixed_questions_dump_to_laya_payload():
    payload = PredictRequest(
        state={"message": "refund me"},
        questions={
            "intent": {
                "type": "choice",
                "instructions": "What intent?",
                "criteria": {"refund": "money back", "other": "anything else"},
            },
            "urgent": {
                "type": "noul",
                "instructions": "Is this urgent?",
            },
        },
    )
    assert payload.questions_payload()["urgent"] == {
        "type": "noul",
        "instructions": "Is this urgent?",
    }
