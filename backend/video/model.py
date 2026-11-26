"""Lazy CLIP zero-shot classifier. Heavy ML dependencies are optional until first inference."""
from __future__ import annotations
import threading
from dataclasses import dataclass
from . import config

_lock = threading.Lock()
_cached = None

@dataclass
class ClipRuntime:
    processor: object
    model: object
    torch: object
    device: object
    source: str = "pretrained"
    version: str = "openai/clip-vit-base-patch32"
    prompt_set: str = "v1"


def _load():
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise RuntimeError("Video model dependencies are missing. Install backend/video-requirements.txt") from exc
    source_mode = config.model_source()
    source, version, prompt_set = "pretrained", config.model_id(), "v1"
    if source_mode in ("auto", "registry"):
        from .registry import RegistryUnavailable, load_champion_config
        try:
            registered = load_champion_config()
            version = registered["version"]
            source = registered["source"]
            prompt_set = registered.get("prompt_set", "v1")
            model_id = registered["model_id"]
        except RegistryUnavailable:
            if source_mode == "registry":
                raise
            model_id = config.model_id()
    else:
        model_id = config.model_id()
    processor = CLIPProcessor.from_pretrained(model_id, cache_dir=str(config.cache_dir()))
    model = CLIPModel.from_pretrained(model_id, cache_dir=str(config.cache_dir()))
    model.eval()
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model.to(device)
    return ClipRuntime(processor, model, torch, device=device, source=source, version=version, prompt_set=prompt_set)


def load_model():
    global _cached
    if _cached is None:
        with _lock:
            if _cached is None:
                _cached = _load()
    return _cached


def reload_model():
    global _cached
    with _lock:
        _cached = None
    return model_info()


def classify_frames(frame_bytes: list[bytes], top_k: int = 3) -> dict:
    if not frame_bytes:
        raise ValueError("At least one extracted video frame is required")
    from io import BytesIO
    from PIL import Image
    runtime = load_model()
    images = [Image.open(BytesIO(blob)).convert("RGB") for blob in frame_bytes]
    inputs = runtime.processor(text=config.PROMPTS, images=images, return_tensors="pt", padding=True).to(runtime.device)
    with runtime.torch.inference_mode():
        outputs = runtime.model(**inputs)
        # Per-frame probabilities across the fixed category taxonomy.
        probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()
    # Mean probability across sampled frames; this is a simple documented baseline.
    mean_scores = probs.mean(axis=0)
    ranked = sorted(zip(config.CATEGORIES, mean_scores.tolist()), key=lambda pair: pair[1], reverse=True)
    best, confidence = ranked[0]
    threshold = config.unknown_threshold()
    category = best if confidence >= threshold else "unknown"
    return {
        "category": category,
        "confidence": float(confidence),
        "best_guess": best,
        "top_k": [{"category": name, "confidence": round(float(score), 4)} for name, score in ranked[:top_k]],
        "reason": "ok" if category != "unknown" else "low_confidence",
        "source": runtime.source,
        "model_version": runtime.version,
    }


def model_info() -> dict:
    runtime = _cached
    return {
        "name": config.registered_model_name(),
        "version": runtime.version if runtime else config.model_id(),
        "source": runtime.source if runtime else "not_loaded",
        "device": str(runtime.device) if runtime else "not_loaded",
        "categories": config.CATEGORIES + ["Other / unknown"],
        "unknown_threshold": config.unknown_threshold(),
        "real_data_validated": False,
        "evaluation_note": "Pretrained CLIP zero-shot baseline; category-level performance has not yet been validated on a representative video dataset.",
        "metrics": {},
        "loaded": runtime is not None,
        "registry_mode": config.model_source(),
    }
