from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

StateInput = str | dict[str, Any] | list[Any]
QuestionType = Literal["choice", "score", "noul"]
PresetName = Literal["guard", "moderation", "triage", "router", "email"]


class QuestionDefinition(BaseModel):
    type: QuestionType
    instructions: str = Field(min_length=1)
    criteria: dict[str, str | None] | list[str] | None = None

    @model_validator(mode="after")
    def validate_criteria(self) -> "QuestionDefinition":
        if self.type == "choice":
            if not isinstance(self.criteria, (dict, list)) or len(self.criteria) < 2:
                raise ValueError("choice questions require at least two criteria")
        elif self.type == "score":
            if not isinstance(self.criteria, list) or len(self.criteria) < 2:
                raise ValueError("score questions require an ordered list with at least two levels")
        return self


class PredictRequest(BaseModel):
    state: StateInput
    questions: dict[str, QuestionDefinition] = Field(min_length=1)
    model: str | None = None
    task: str | None = None
    lang: str | None = None

    def questions_payload(self) -> dict[str, dict[str, Any]]:
        return {name: question.model_dump(exclude_none=True) for name, question in self.questions.items()}


class RouteRequest(BaseModel):
    state: StateInput
    questions: dict[str, QuestionDefinition] | None = None
    model: str | None = None
    task: str | None = None
    lang: str | None = None

    def questions_payload(self) -> dict[str, dict[str, Any]] | None:
        if self.questions is None:
            return None
        return {name: question.model_dump(exclude_none=True) for name, question in self.questions.items()}


class PresetPredictRequest(BaseModel):
    state: StateInput
    model: str | None = None
    task: str | None = None
    lang: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    loaded_models: list[str]
