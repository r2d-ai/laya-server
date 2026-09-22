import asyncio
import logging
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


class LayaRuntime:
    def __init__(self) -> None:
        self._router: Any | None = None
        self._laya: Any | None = None
        self._torch: Any | None = None
        self._semaphore: asyncio.Semaphore | None = None

    @property
    def ready(self) -> bool:
        return self._router is not None

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

        preload = settings.preload_models
        router = laya.Router(
            device=settings.device,
            token=settings.hf_token,
            max_loaded=max(settings.max_loaded, len(preload) or 1),
            default=settings.default_model,
            auto_task_detection=settings.auto_task_detection,
            preload=False,
        )
        if preload:
            logger.info("Preloading Laya models on %s: %s", settings.device, ", ".join(preload))
            router.preload(preload)

        self._laya = laya
        self._torch = torch
        self._router = router
        self._semaphore = asyncio.Semaphore(settings.max_concurrency)

        if settings.strict_cuda:
            for name in list(router.loaded):
                self._assert_agent_device(router.load(name), name)

        logger.info("Laya runtime ready; loaded=%s", router.loaded)

    async def start_async(self) -> None:
        await asyncio.to_thread(self.start)

    def stop(self) -> None:
        if self._router is not None:
            self._router.unload()
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
        self._router = None
        self._laya = None
        self._torch = None
        self._semaphore = None

    def _require_router(self) -> Any:
        if self._router is None:
            raise RuntimeError("Laya runtime is not initialized")
        return self._router

    def _assert_agent_device(self, agent: Any, model_name: str) -> None:
        settings = get_settings()
        if settings.strict_cuda and getattr(getattr(agent, "device", None), "type", None) != "cuda":
            raise RuntimeError(
                f"Laya model {model_name!r} fell back to CPU while LAYA_STRICT_CUDA=true"
            )

    def _predict_sync(
        self,
        state: Any,
        questions: dict[str, Any],
        model: str | None,
        task: str | None,
        lang: str | None,
    ) -> dict[str, Any]:
        router = self._require_router()
        decision = router.route(state, questions, model=model, task=task, lang=lang)
        agent = router.load(decision["model"])
        self._assert_agent_device(agent, decision["model"])
        result = agent.system_one(state, questions)
        result["routing"] = dict(decision)
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
            raise RuntimeError("Laya runtime is not initialized")
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
        router = self._require_router()
        decision = await asyncio.to_thread(
            router.route, state, questions, model, task, lang
        )
        return dict(decision)

    def preset_questions(self, preset: str) -> dict[str, Any]:
        if self._laya is None:
            raise RuntimeError("Laya runtime is not initialized")
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
        router = self._require_router()
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
            "laya_version": getattr(laya, "__version__", None),
            "torch_version": getattr(torch, "__version__", None),
            "device": settings.device,
            "strict_cuda": settings.strict_cuda,
            "cuda_available": cuda_available,
            "gpu": gpu,
            "loaded_models": list(router.loaded),
            "preload_models": settings.preload_models,
            "max_loaded": router.max_loaded,
            "max_concurrency": settings.max_concurrency,
            "default_model": router.default,
            "auto_task_detection": router.auto_task_detection,
            "presets": sorted(_PRESET_BUILDERS),
        }


runtime = LayaRuntime()
