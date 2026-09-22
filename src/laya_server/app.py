from contextlib import asynccontextmanager
import secrets

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastmcp.utilities.lifespan import combine_lifespans

from .api import router as api_router
from .mcp import mcp
from .runtime import runtime
from .settings import get_settings


@asynccontextmanager
async def runtime_lifespan(_: FastAPI):
    await runtime.start_async()
    try:
        yield
    finally:
        runtime.stop()


mcp_app = mcp.http_app(path="/")
app = FastAPI(
    title="Laya Server",
    summary="CUDA-backed Laya REST/OpenAPI and MCP service",
    description=(
        "Serves NandhaKishorM/laya through a regular REST/OpenAPI API and a curated FastMCP endpoint. "
        "Both interfaces share the same resident CUDA models."
    ),
    version="0.1.0",
    lifespan=combine_lifespans(runtime_lifespan, mcp_app.lifespan),
)


@app.middleware("http")
async def optional_api_key(request: Request, call_next):
    settings = get_settings()
    expected = settings.api_key
    if not expected or request.url.path in {"/healthz", "/readyz", "/docs", "/openapi.json", "/redoc"}:
        return await call_next(request)

    supplied = request.headers.get("x-api-key")
    authorization = request.headers.get("authorization", "")
    if not supplied and authorization.lower().startswith("bearer "):
        supplied = authorization[7:].strip()

    if not supplied or not secrets.compare_digest(supplied, expected):
        return JSONResponse(status_code=401, content={"detail": "invalid or missing API key"})
    return await call_next(request)


app.include_router(api_router)
app.mount("/mcp", mcp_app)
