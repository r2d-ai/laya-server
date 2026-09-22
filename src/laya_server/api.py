import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Response, status

from .runtime import runtime
from .schemas import (
    HealthResponse,
    PredictRequest,
    PresetName,
    PresetPredictRequest,
    ReadyResponse,
    RouteRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, ValueError):
        return HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    if isinstance(exc, RuntimeError):
        return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    logger.exception("Laya request failed")
    return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Laya inference failed")


@router.get("/healthz", response_model=HealthResponse, operation_id="health")
async def health() -> HealthResponse:
    return HealthResponse()


@router.get("/readyz", response_model=ReadyResponse, operation_id="readiness")
async def readiness(response: Response) -> ReadyResponse:
    if not runtime.ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadyResponse(status="not_ready", loaded_models=[])
    return ReadyResponse(status="ready", loaded_models=runtime.info()["loaded_models"])


@router.get("/v1/info", operation_id="laya_info")
async def info() -> dict[str, Any]:
    try:
        return runtime.info()
    except Exception as exc:
        raise _http_error(exc) from exc


@router.post("/v1/route", operation_id="laya_route")
async def route_laya(payload: RouteRequest) -> dict[str, Any]:
    try:
        return await runtime.route(
            state=payload.state,
            questions=payload.questions_payload(),
            model=payload.model,
            task=payload.task,
            lang=payload.lang,
        )
    except Exception as exc:
        raise _http_error(exc) from exc


@router.post("/v1/predict", operation_id="laya_predict")
async def predict_laya(payload: PredictRequest) -> dict[str, Any]:
    try:
        return await runtime.predict(
            state=payload.state,
            questions=payload.questions_payload(),
            model=payload.model,
            task=payload.task,
            lang=payload.lang,
        )
    except Exception as exc:
        raise _http_error(exc) from exc


@router.get("/v1/presets", operation_id="laya_list_presets")
async def list_presets() -> dict[str, dict[str, Any]]:
    try:
        return {name: runtime.preset_questions(name) for name in runtime.info()["presets"]}
    except Exception as exc:
        raise _http_error(exc) from exc


@router.post("/v1/presets/{preset}/predict", operation_id="laya_predict_preset")
async def predict_preset(preset: PresetName, payload: PresetPredictRequest) -> dict[str, Any]:
    try:
        return await runtime.predict_preset(
            preset=preset,
            state=payload.state,
            model=payload.model,
            task=payload.task,
            lang=payload.lang,
        )
    except Exception as exc:
        raise _http_error(exc) from exc
