# laya-server

CUDA-backed service wrapper for [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya), exposing the same resident Laya runtime through:

- **REST + OpenAPI** via FastAPI (`/v1/*`, docs at `/docs`)
- **MCP over HTTP** via FastMCP (`/mcp/`)
- **CUDA-first production startup** with explicit failure instead of silent CPU fallback
- **Shared model residency** so REST and MCP do not duplicate VRAM

Laya's router can dispatch English, multilingual, and typed-decisions checkpoints. This server preloads all three by default for predictable latency; change `LAYA_PRELOAD` if the GPU does not have enough VRAM.

## Run with Docker + NVIDIA GPU

Requirements: Docker, Docker Compose, NVIDIA Container Toolkit, and a supported NVIDIA driver.

```bash
docker compose up -d --build
curl http://localhost:8000/readyz
curl http://localhost:8000/v1/info
```

The image uses CUDA 12.6 and PyTorch 2.14's `cu126` wheel. Models are cached in the `laya-hf-cache` volume.

## Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /healthz` | process liveness |
| `GET /readyz` | model/runtime readiness |
| `GET /v1/info` | CUDA, GPU, Laya and loaded-model information |
| `POST /v1/route` | inspect which checkpoint Laya would select, without inference |
| `POST /v1/predict` | custom typed decision request |
| `GET /v1/presets` | built-in Laya question schemas |
| `POST /v1/presets/{preset}/predict` | run `guard`, `moderation`, `triage`, `router`, or `email` preset |
| `GET /docs` | Swagger UI / OpenAPI |
| `/mcp/` | FastMCP HTTP endpoint |

### REST example

```bash
curl -s http://localhost:8000/v1/predict \
  -H 'content-type: application/json' \
  -d '{
    "state": {"message": "Refund the duplicate charge today"},
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "Which department should handle this?",
        "criteria": {
          "billing": "payments, invoices, refunds",
          "technical": "bugs or outages",
          "other": "everything else"
        }
      },
      "urgent": {
        "type": "noul",
        "instructions": "Is this time-sensitive?"
      }
    }
  }'
```

### Preset example

```bash
curl -s http://localhost:8000/v1/presets/router/predict \
  -H 'content-type: application/json' \
  -d '{"state":{"request":"Refactor this service using dependency injection"}}'
```

## MCP tools

The MCP endpoint intentionally exposes a small, agent-friendly surface instead of mechanically mirroring every REST endpoint:

- `laya_predict` — custom `choice`, `score`, and `noul` decisions
- `laya_route` — checkpoint routing decision only
- `laya_predict_preset` — built-in Laya workflows
- `laya_get_preset` — inspect a preset schema
- `laya_info` — runtime/GPU information

Example MCP config for a client that supports remote HTTP servers:

```json
{
  "mcpServers": {
    "laya": {
      "url": "http://localhost:8000/mcp/"
    }
  }
}
```

If `LAYA_API_KEY` is set, pass either `Authorization: Bearer <key>` or `X-API-Key: <key>` to REST and MCP. Liveness, readiness and API docs remain unauthenticated.

## Configuration

| Environment variable | Default | Notes |
|---|---|---|
| `LAYA_DEVICE` | `cuda` | passed to `laya.Router` |
| `LAYA_STRICT_CUDA` | `true` | fail startup/inference if CUDA is unavailable or Laya falls back to CPU |
| `LAYA_PRELOAD` | `english,multilingual,typed-decisions` | comma-separated checkpoints; set empty for lazy loading |
| `LAYA_MAX_LOADED` | `3` | Laya router resident model cap |
| `LAYA_DEFAULT_MODEL` | `english` | router fallback |
| `LAYA_AUTO_TASK_DETECTION` | `true` | allow exact typed-decisions workflow detection |
| `LAYA_MAX_CONCURRENCY` | `2` | bounds concurrent GPU inference from REST + MCP combined |
| `LAYA_API_KEY` | unset | optional shared API key |
| `LAYA_HF_TOKEN` | unset | Hugging Face token passed to Laya |
| `LAYA_HOST` | `0.0.0.0` | bind address |
| `LAYA_PORT` | `8000` | bind port |

For CPU-only local development, set `LAYA_DEVICE=cpu` and `LAYA_STRICT_CUDA=false`.

## Notes on GPU memory

Upstream Laya documents three checkpoints totalling roughly 1.16B parameters. Preloading all models removes language-switch reload latency but consumes more VRAM. If the GPU is constrained, start with:

```bash
LAYA_PRELOAD=english,multilingual LAYA_MAX_LOADED=2 docker compose up -d
```

or lazy-load one model at a time:

```bash
LAYA_PRELOAD= LAYA_MAX_LOADED=1 docker compose up -d
```

## Local development

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
LAYA_DEVICE=cpu LAYA_STRICT_CUDA=false LAYA_PRELOAD= laya-server
```

Run tests:

```bash
pytest
```
