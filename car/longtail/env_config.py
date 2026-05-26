import os
import urllib.request
from pathlib import Path
from typing import Any, Dict, List


MODULE_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODULE_DIR.parents[1]
DEFAULT_MODEL_DIR = MODULE_DIR / "models"

DEFAULT_CLIP_LABELS = [
    "road construction with traffic cones",
    "traffic cones on road",
    "orange safety cones blocking the way",
    "construction zone with cones",
    "normal road driving scene",
    "ordinary street scene",
    "highway driving scene",
    "pedestrian",
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "traffic cone",
    "traffic bollard",
    "road barrier",
    "barricade",
    "construction sign",
    "detour sign",
    "lane closed sign",
    "temporary traffic sign",
    "warning sign",
    "arrow board",
    "road work",
    "traffic light",
    "stop sign",
]

DEFAULT_YOLOWORLD_CLASSES = [
    "person",
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "scooter",
    "traffic cone",
    "traffic bollard",
    "road barrier",
    "traffic barrier",
    "barricade",
    "construction sign",
    "road work sign",
    "detour sign",
    "lane closed sign",
    "temporary sign",
    "arrow board",
    "warning sign",
]


def load_longtail_config_from_env() -> Dict[str, Any]:
    _load_env_file()
    _apply_runtime_performance_settings()
    detectors = []
    for detector_type in _env_list("CAR_LONGTAIL_DETECTORS", ["clip", "yolov8", "yolopv2"], sep=","):
        detector_type = detector_type.lower()
        if detector_type == "clip":
            detectors.append(_clip_detector_config())
        elif detector_type == "yoloworld":
            detectors.append(_yoloworld_detector_config())
        elif detector_type == "yolov8":
            detectors.append(_yolov8_detector_config())
        elif detector_type == "yolopv2":
            detectors.append(_yolopv2_detector_config())
        else:
            raise ValueError(f"Unsupported detector in CAR_LONGTAIL_DETECTORS: {detector_type}")

    return {
        "threshold": _env_float("CAR_LONGTAIL_THRESHOLD", 0.5),
        "detectors": detectors,
    }


def _clip_detector_config() -> Dict[str, Any]:
    model_name = _resolve_hf_model(
        configured=_env_str("CAR_LONGTAIL_CLIP_MODEL", "/media/pi/FSDBY/weights/clip-vit-base-patch32"),
        remote=_env_str("CAR_LONGTAIL_CLIP_REMOTE_MODEL", "openai/clip-vit-base-patch32"),
        cache_name="clip-vit-base-patch32",
    )
    return {
        "type": "clip",
        "weight": _env_float("CAR_LONGTAIL_CLIP_WEIGHT", 0.5),
        "config": {
            "model_name": model_name,
            "device": _env_str("CAR_LONGTAIL_CLIP_DEVICE", "cpu"),
            "max_prob_threshold": _env_float("CAR_LONGTAIL_CLIP_MAX_PROB_THRESHOLD", 0.22),
            "entropy_threshold": _env_float("CAR_LONGTAIL_CLIP_ENTROPY_THRESHOLD", 0.72),
            "labels": _env_list("CAR_LONGTAIL_CLIP_LABELS", DEFAULT_CLIP_LABELS, sep="|"),
        },
    }


def _yoloworld_detector_config() -> Dict[str, Any]:
    model_path = _resolve_ultralytics_model(
        configured=_env_str("CAR_LONGTAIL_YOLOWORLD_MODEL", "/media/pi/FSDBY/weights/yolov8x-worldv2.pt"),
        remote=_env_str("CAR_LONGTAIL_YOLOWORLD_REMOTE_MODEL", "yolov8x-worldv2.pt"),
        remote_url=_env_str("CAR_LONGTAIL_YOLOWORLD_REMOTE_URL", ""),
        cache_name="yolov8x-worldv2.pt",
    )
    return {
        "type": "yoloworld",
        "weight": _env_float("CAR_LONGTAIL_YOLOWORLD_WEIGHT", 0.3),
        "config": {
            "model_path": model_path,
            "conf_threshold": _env_float("CAR_LONGTAIL_YOLOWORLD_CONF_THRESHOLD", 0.20),
            "classes": _env_list("CAR_LONGTAIL_YOLOWORLD_CLASSES", DEFAULT_YOLOWORLD_CLASSES, sep="|"),
        },
    }


def _yolov8_detector_config() -> Dict[str, Any]:
    model_path = _resolve_ultralytics_model(
        configured=_env_str("CAR_LONGTAIL_YOLOV8_MODEL", "/media/pi/FSDBY/weights/yolov8n.pt"),
        remote=_env_str("CAR_LONGTAIL_YOLOV8_REMOTE_MODEL", "yolov8n.pt"),
        remote_url=_env_str(
            "CAR_LONGTAIL_YOLOV8_REMOTE_URL",
            "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt",
        ),
        cache_name="yolov8n.pt",
    )
    return {
        "type": "yolov8",
        "weight": _env_float("CAR_LONGTAIL_YOLOV8_WEIGHT", 0.1),
        "config": {
            "model_path": model_path,
            "conf_threshold": _env_float("CAR_LONGTAIL_YOLOV8_CONF_THRESHOLD", 0.25),
            "img_size": _env_int("CAR_LONGTAIL_YOLOV8_IMG_SIZE", 320),
            "unusual_indicators": _env_list(
                "CAR_LONGTAIL_YOLOV8_UNUSUAL_INDICATORS",
                ["stop sign", "parking meter", "fire hydrant"],
                sep="|",
            ),
        },
    }


def _yolopv2_detector_config() -> Dict[str, Any]:
    use_full_model = _env_bool("CAR_LONGTAIL_YOLOPV2_USE_FULL_MODEL", False)
    weights_path = _env_str("CAR_LONGTAIL_YOLOPV2_WEIGHTS", "/media/pi/FSDBY/weights/yolopv2.pt")
    if use_full_model:
        weights_path = _resolve_file_model(
            configured=weights_path,
            remote_url=_env_str("CAR_LONGTAIL_YOLOPV2_REMOTE_URL", ""),
            cache_name="yolopv2.pt",
        )
    return {
        "type": "yolopv2",
        "weight": _env_float("CAR_LONGTAIL_YOLOPV2_WEIGHT", 0.1),
        "config": {
            "weights_path": weights_path,
            "img_size": _env_int("CAR_LONGTAIL_YOLOPV2_IMG_SIZE", 640),
            "conf_threshold": _env_float("CAR_LONGTAIL_YOLOPV2_CONF_THRESHOLD", 0.3),
            "use_full_model": use_full_model,
            "device": _env_str("CAR_LONGTAIL_YOLOPV2_DEVICE", "cpu"),
            "geometry_mode": _env_str("CAR_LONGTAIL_YOLOPV2_GEOMETRY_MODE", "weighted"),
            "fast_mask": _env_bool("CAR_LONGTAIL_YOLOPV2_FAST_MASK", True),
            "geometry_weights": {
                "near_missing": _env_float("CAR_LONGTAIL_YOLOPV2_NEAR_MISSING_WEIGHT", 0.22),
                "center_missing": _env_float("CAR_LONGTAIL_YOLOPV2_CENTER_MISSING_WEIGHT", 0.18),
                "width_jump": _env_float("CAR_LONGTAIL_YOLOPV2_WIDTH_JUMP_WEIGHT", 0.24),
                "center_jump": _env_float("CAR_LONGTAIL_YOLOPV2_CENTER_JUMP_WEIGHT", 0.16),
                "lane_anomaly": _env_float("CAR_LONGTAIL_YOLOPV2_LANE_ANOMALY_WEIGHT", 0.10),
                "width_std": _env_float("CAR_LONGTAIL_YOLOPV2_WIDTH_STD_WEIGHT", 0.10),
            },
            "linear_coefficients": _env_float_list(
                "CAR_LONGTAIL_YOLOPV2_LINEAR_COEFFICIENTS",
                [0.408258, 0.611690, 0.900184, 0.261055, 0.375240, -2.352794, 6.273399, -2.959277, -1.691969, -3.064723],
                sep="|",
            ),
            "linear_intercept": _env_float("CAR_LONGTAIL_YOLOPV2_LINEAR_INTERCEPT", 0.554888),
            "linear_temperature": _env_float("CAR_LONGTAIL_YOLOPV2_LINEAR_TEMPERATURE", 1.0),
        },
    }


def _resolve_hf_model(configured: str, remote: str, cache_name: str) -> str:
    path = _expand_path(configured)
    if _looks_like_path(configured) and path.exists():
        return str(path)
    if not _looks_like_path(configured):
        return configured
    if not _env_bool("CAR_LONGTAIL_AUTO_DOWNLOAD", True):
        return configured

    local_dir = _model_dir() / cache_name
    if local_dir.exists() and any(local_dir.iterdir()):
        return str(local_dir)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to auto-download CLIP models") from exc

    print(f"Model path not found: {configured}. Downloading {remote} to {local_dir}")
    return snapshot_download(repo_id=remote, local_dir=str(local_dir))


def _resolve_ultralytics_model(configured: str, remote: str, remote_url: str, cache_name: str) -> str:
    path = _expand_path(configured)
    if _looks_like_path(configured) and path.exists():
        return str(path)
    if not _looks_like_path(configured):
        return configured
    if not _env_bool("CAR_LONGTAIL_AUTO_DOWNLOAD", True):
        return configured
    if remote_url:
        return _resolve_file_model(configured, remote_url, cache_name)
    print(f"Model path not found: {configured}. Using Ultralytics auto-download model: {remote}")
    return remote


def _resolve_file_model(configured: str, remote_url: str, cache_name: str) -> str:
    path = _expand_path(configured)
    if path.exists():
        return str(path)
    if not remote_url or not _env_bool("CAR_LONGTAIL_AUTO_DOWNLOAD", True):
        return configured

    local_path = _model_dir() / cache_name
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if not local_path.exists():
        print(f"Model path not found: {configured}. Downloading {remote_url} to {local_path}")
        urllib.request.urlretrieve(remote_url, local_path)
    return str(local_path)


def _model_dir() -> Path:
    return _expand_path(_env_str("CAR_LONGTAIL_MODEL_DIR", str(DEFAULT_MODEL_DIR)))


def _apply_runtime_performance_settings() -> None:
    torch_threads = _env_int("CAR_LONGTAIL_TORCH_NUM_THREADS", 0)
    if torch_threads > 0:
        try:
            import torch

            torch.set_num_threads(torch_threads)
            interop_threads = _env_int("CAR_LONGTAIL_TORCH_INTEROP_THREADS", 1)
            if interop_threads > 0:
                try:
                    torch.set_num_interop_threads(interop_threads)
                except RuntimeError:
                    pass
        except ImportError:
            pass

    opencv_threads = _env_int("CAR_LONGTAIL_OPENCV_NUM_THREADS", 0)
    if opencv_threads > 0:
        try:
            import cv2

            cv2.setNumThreads(opencv_threads)
        except ImportError:
            pass


def _load_env_file() -> None:
    configured = os.environ.get("CAR_LONGTAIL_ENV_FILE", "").strip()
    candidates = [Path(configured)] if configured else [REPO_ROOT / ".env", REPO_ROOT / ".env_example"]
    env_file = next((path for path in candidates if path.exists()), None)
    if env_file is None:
        return
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError("python-dotenv is required to load .env files") from exc
    load_dotenv(env_file, override=False)


def _expand_path(value: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(value)))


def _looks_like_path(value: str) -> bool:
    return value.startswith(("/", ".", "~")) or "\\" in value or value.count("/") > 1


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None or value.strip() == "" else float(value)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None or value.strip() == "" else int(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: List[str], sep: str) -> List[str]:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return list(default)
    return [item.strip() for item in value.split(sep) if item.strip()]


def _env_float_list(name: str, default: List[float], sep: str) -> List[float]:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return list(default)
    return [float(item.strip()) for item in value.split(sep) if item.strip()]
