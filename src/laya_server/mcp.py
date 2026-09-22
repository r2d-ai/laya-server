from typing import Any

from fastmcp import FastMCP

from .runtime import runtime
from .schemas import PredictRequest, PresetName, PresetPredictRequest, RouteRequest

mcp = FastMCP(
    "Laya Decision Engine",
    instructions=(
        "Fast non-autoregressive typed decisions. Use laya_route to inspect checkpoint routing, "
        "laya_predict for custom choice/score/noul questions, and laya_predict_preset for common workflows."
    ),
)


@mcp.tool(name="laya_predict")
async def laya_predict(
    state: str | dict[str, Any] | list[Any],
    questions: dict[str, dict[str, Any]],
    model: str | None = None,
    task: str | None = None,
    lang: str | None = None,
) -> dict[str, Any]:
    """Run custom typed Laya decisions in one model forward pass."""
    payload = PredictRequest(state=state, questions=questions, model=model, task=task, lang=lang)
    return await runtime.predict(
        state=payload.state,
        questions=payload.questions_payload(),
        model=payload.model,
        task=payload.task,
        lang=payload.lang,
    )


@mcp.tool(name="laya_route")
async def laya_route(
    state: str | dict[str, Any] | list[Any],
    questions: dict[str, dict[str, Any]] | None = None,
    model: str | None = None,
    task: str | None = None,
    lang: str | None = None,
) -> dict[str, Any]:
    """Choose the Laya checkpoint for a request without running inference."""
    payload = RouteRequest(state=state, questions=questions, model=model, task=task, lang=lang)
    return await runtime.route(
        state=payload.state,
        questions=payload.questions_payload(),
        model=payload.model,
        task=payload.task,
        lang=payload.lang,
    )


@mcp.tool(name="laya_predict_preset")
async def laya_predict_preset(
    preset: PresetName,
    state: str | dict[str, Any] | list[Any],
    model: str | None = None,
    task: str | None = None,
    lang: str | None = None,
) -> dict[str, Any]:
    """Run one of Laya's built-in guard, moderation, triage, router, or email presets."""
    payload = PresetPredictRequest(state=state, model=model, task=task, lang=lang)
    return await runtime.predict_preset(
        preset=preset,
        state=payload.state,
        model=payload.model,
        task=payload.task,
        lang=payload.lang,
    )


@mcp.tool(name="laya_get_preset")
async def laya_get_preset(preset: PresetName) -> dict[str, Any]:
    """Return the exact question schema used by a built-in Laya preset."""
    return runtime.preset_questions(preset)


@mcp.tool(name="laya_info")
async def laya_info() -> dict[str, Any]:
    """Return runtime, CUDA, checkpoint, and concurrency information."""
    return runtime.info()
