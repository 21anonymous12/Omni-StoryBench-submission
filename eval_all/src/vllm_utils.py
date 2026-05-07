import gc
import importlib.util
import json
from typing import Any, Dict, Iterable


_VLLM_LLM_CACHE: Dict[str, Any] = {}


def is_vllm_available() -> bool:
    return importlib.util.find_spec("vllm") is not None


def resolve_backend(requested_backend: str) -> str:
    backend = requested_backend.lower()
    if backend not in {"auto", "transformers", "vllm"}:
        raise ValueError(
            f"Unsupported backend: {requested_backend}. Expected one of auto/transformers/vllm."
        )

    if backend == "auto":
        return "vllm" if is_vllm_available() else "transformers"

    if backend == "vllm" and not is_vllm_available():
        raise RuntimeError(
            "The vLLM backend was requested, but the `vllm` package is not installed "
            "in the current Python environment."
        )

    return backend


def should_enable_expert_parallel(model_name: str) -> bool:
    normalized = model_name.lower()
    return any(token in normalized for token in ("a3b", "moe"))


def _make_cacheable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _make_cacheable(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_make_cacheable(item) for item in value]
    return value


def _build_cache_key(model_name: str, engine_kwargs: Dict[str, Any]) -> str:
    payload = {
        "model_name": model_name,
        "engine_kwargs": _make_cacheable(engine_kwargs),
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=True)


def get_vllm_llm(model_name: str, **engine_kwargs: Any) -> Any:
    resolve_backend("vllm")
    filtered_kwargs = {key: value for key, value in engine_kwargs.items() if value is not None}
    cache_key = _build_cache_key(model_name, filtered_kwargs)

    cached = _VLLM_LLM_CACHE.get(cache_key)
    if cached is not None:
        return cached

    from vllm import LLM

    llm = LLM(model=model_name, **filtered_kwargs)
    _VLLM_LLM_CACHE[cache_key] = llm
    return llm


def _iter_matching_cache_keys(model_name: str | None = None) -> Iterable[str]:
    for cache_key in list(_VLLM_LLM_CACHE):
        if model_name is None:
            yield cache_key
            continue
        payload = json.loads(cache_key)
        if payload.get("model_name") == model_name:
            yield cache_key


def force_memory_cleanup() -> None:
    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass


def _shutdown_vllm_llm(llm: Any) -> None:
    llm_engine = getattr(llm, "llm_engine", None)
    if llm_engine is None:
        return

    engine_core = getattr(llm_engine, "engine_core", None)
    if engine_core is not None and hasattr(engine_core, "shutdown"):
        engine_core.shutdown()

    renderer = getattr(llm_engine, "renderer", None)
    if renderer is not None and hasattr(renderer, "shutdown"):
        renderer.shutdown()

    try:
        from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

        cleanup_dist_env_and_memory()
    except Exception:
        pass


def release_vllm_models(model_name: str | None = None) -> None:
    released_any = False
    for cache_key in list(_iter_matching_cache_keys(model_name=model_name)):
        llm = _VLLM_LLM_CACHE.pop(cache_key, None)
        if llm is None:
            continue
        released_any = True
        try:
            _shutdown_vllm_llm(llm)
        except Exception:
            pass
        del llm

    if released_any:
        force_memory_cleanup()
