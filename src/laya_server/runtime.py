import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from .settings import get_settings

logger = logging.getLogger(__name__)

_PRESET_BUILDERS = {
    "guard": "guard_questions",
    "moderation": "moderation_questions",
    "triage": "triage_questions",
    "router": "router_questions",
    "email": "email_questions",
}

_MODEL_NAME = "multilingual"
_MODEL_REPO = "convaiinnovations/laya"
_MODEL_SUBFOLDER = "multilingual"

_WARMUP_STATE = {"text": "warmup"}
_WARMUP_QUESTIONS = {
    "ready": {
        "type": "noul",
        "instructions": "Is this a warmup request?",
    }
}


def _candidate_laya_snapshots() -> list[Path]:
    """Return cached convaiinnovations/laya snapshots, preferring refs/main."""
    hf_home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    repo_cache = hf_home / "hub" / "models--convaiinnovations--laya"
    snapshots = repo_cache / "snapshots"
    if not snapshots.is_dir():
        return []

    candidates: list[Path] = []
    main_ref = repo_cache / "refs" / "main"
    if main_ref.is_file():
        try:
            sha = main_ref.read_text(encoding="utf-8").strip()
            preferred = snapshots / sha
            if preferred.is_dir():
                candidates.append(preferred)
        except OSError:
            pass

    try:
        others = sorted(
            (p for p in snapshots.iterdir() if p.is_dir() and p not in candidates),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        others = []
    candidates.extend(others)
    return candidates


def _model_cached(snapshot: Path) -> bool:
    root = snapshot / _MODEL_SUBFOLDER
    required = (
        root / "rl_agent_config.json",
        root / "model.safetensors",
        root / "tokenizer",
        root / "encoder",
    )
    return all(path.exists() for path in required)


def _cached_multilingual_snapshot() -> Path | None:
    for snapshot in _candidate_laya_snapshots():
        if _model_cached(snapshot):
            return snapshot
    return None


class LayaRuntime:
    def __init__(self) -> None:
        self._agent: Any | None = None
        self._laya: Any | None = None
        self._torch: Any | None = None
        self._semaphore: asyncio.Semaphore | None = None
        self._cached = False
        self._warmed = False

    @property
    def ready(self) -> bool:
        return self._agent is not None and self._warmed

    def _validate_model(self, model: str | None) -> None:
        if model is not None and model.strip().lower() not in {
            "multilingual",
            "multi",
            "ml",
            "laya-multilingual",
        }:
            raise ValueError(
                f"this server only provides the multilingual checkpoint; got model={model!r}"
            )

    def _assert_agent_device(self, agent: Any) -> None:
        settings = get_settings()
        if settings.strict_cuda and getattr(getattr(agent, "device", None), "type", None) != "cuda":
            raise RuntimeError(
                "Laya multilingual checkpoint fell back to CPU while LAYA_STRICT_CUDA=true"
            )

    def start(self) -> None:
        if self.ready:
            return

        import laya
        import torch

        settings = get_settings()
        if settings.device.startswith("cuda") and settings.strict_cuda and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required (LAYA_STRICT_CUDA=true) but torch.cuda.is_available() is false"
            )

        self._laya = laya
        self._torch = torch

        snapshot = _cached_multilingual_snapshot()
        if snapshot is not None:
            self._cached = True
            model_source = str(snapshot)
            logger.info("Loading multilingual checkpoint from persistent HF cache: %s", snapshot)
        else:
            self._cached = False
            model_source = _MODEL_REPO
            logger.info(
                "Multilingual checkpoint not found in cache; downloading %s/%s",
                _MODEL_REPO,
                _MODEL_SUBFOLDER,
            )

        logger.info("Loading Laya multilingual checkpoint on %s", settings.device)
        agent = laya.Agent(
            model_source,
            device=settings.device,
            token=settings.hf_token,
            subfolder=_MODEL_SUBFOLDER,
        )
        self._assert_agent_device(agent)
        logger.info("Laya multilingual checkpoint loaded; starting CUDA/Triton warmup")

        agent.system_one(_WARMUP_STATE, _WARMUP_QUESTIONS)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self._agent = agent
        self._warmed = True
        self._semaphore = asyncio.Semaphore(settings.max_concurrency)
        logger.info("Laya multilingual runtime ready")

    async def start_async(self) -> None:
        await asyncio.to_thread(self.start)

    def stop(self) -> None:
        self._agent = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
        self._laya = None
        self._torch = None
        self._semaphore = None
        self._cached = False
        self._warmed = False

    def _require_agent(self) -> Any:
        if self._agent is None:
            raise RuntimeError("Laya multilingual runtime is not initialized")
        return self._agent

    def _predict_sync(
        self,
        state: Any,
        questions: dict[str, Any],
        model: str | None,
        task: str | None,
        lang: str | None,
    ) -> dict[str, Any]:
        del task, lang
        self._validate_model(model)
        agent = self._require_agent()
        self._assert_agent_device(agent)
        result = agent.system_one(state, questions)
        result["routing"] = {
            "model": _MODEL_NAME,
            "repo": f"{_MODEL_REPO}/{_MODEL_SUBFOLDER}",
            "reason": "single-model server: multilingual checkpoint is always used",
            "detection": None,
            "workflow": None,
        }
        return result

    async def predict(
        self,
        state: Any,
        questions: dict[str, Any],
        model: str | None = None,
        task: str | None = None,
        lang: str | None = None,
    ) -> dict[str, Any]:
        if self._semaphore is None:
            raise RuntimeError("Laya multilingual runtime is not initialized")
        async with self._semaphore:
            return await asyncio.to_thread(
                self._predict_sync, state, questions, model, task, lang
            )

    async def route(
        self,
        state: Any,
        questions: dict[str, Any] | None = None,
        model: str | None = None,
        task: str | None = None,
        lang: str | None = None,
    ) -> dict[str, Any]:
        del state, questions, task, lang
        self._validate_model(model)
        return {
            "model": _MODEL_NAME,
            "repo": f"{_MODEL_REPO}/{_MODEL_SUBFOLDER}",
            "reason": "single-model server: multilingual checkpoint is always used",
            "detection": None,
            "workflow": None,
        }

    def preset_questions(self, preset: str) -> dict[str, Any]:
        if self._laya is None:
            raise RuntimeError("Laya multilingual runtime is not initialized")
        try:
            builder_name = _PRESET_BUILDERS[preset]
        except KeyError as exc:
            raise ValueError(f"unknown preset {preset!r}; choose one of {sorted(_PRESET_BUILDERS)}") from exc
        return getattr(self._laya, builder_name)()

    async def predict_preset(
        self,
        preset: str,
        state: Any,
        model: str | None = None,
        task: str | None = None,
        lang: str | None = None,
    ) -> dict[str, Any]:
        return await self.predict(
            state=state,
            questions=self.preset_questions(preset),
            model=model,
            task=task,
            lang=lang,
        )

    def info(self) -> dict[str, Any]:
        agent = self._require_agent()
        torch = self._torch
        laya = self._laya
        settings = get_settings()

        cuda_available = bool(torch and torch.cuda.is_available())
        gpu: dict[str, Any] | None = None
        if cuda_available:
            index = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(index)
            gpu = {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
                "total_memory_bytes": props.total_memory,
            }

        return {
            "service": "laya-server",
            "mode": "multilingual-only",
            "laya_version": getattr(laya, "__version__", None),
            "torch_version": getattr(torch, "__version__", None),
            "device": str(getattr(agent, "device", settings.device)),
            "strict_cuda": settings.strict_cuda,
            "cuda_available": cuda_available,
            "gpu": gpu,
            "loaded_models": [_MODEL_NAME],
            "cached_models": [_MODEL_NAME] if self._cached else [],
            "warmed_models": [_MODEL_NAME] if self._warmed else [],
            "max_concurrency": settings.max_concurrency,
            "presets": sorted(_PRESET_BUILDERS),
        }


runtime = LayaRuntime()
